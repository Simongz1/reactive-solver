// Three phases with a common velocity and temperature, and volume-weighted pressure.
// Deviatoric stress and plasticity are not included. Pores contain ideal-gas air.
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "multimaterialEOS.h"
#include "computeReactionRates.h"
#include "densityMap2D.h"
#include "levelSet2D.h"
#include "WENO.h"
#include "THINC.h"
#include "ExodusOutput.h"

using Grid = levelSet2D::Grid;
enum Field
{
    Species1Mass = 0,
    Species2Mass,
    Species3Mass,
    AlphaRhoRDX,
    AlphaRhoPS,
    AlphaRhoPore,
    MomentumX,
    MomentumY,
    TotalEnergy,
    AlphaRDX,
    AlphaPS,
    AlphaPore
};
constexpr int conservedFieldCount = 9;
constexpr int stateFieldCount = 12;
using Cell = Eigen::Matrix<double, stateFieldCount, 1>;
using State = Eigen::Matrix<double, stateFieldCount, Eigen::Dynamic>;
using Conserved = Eigen::Matrix<double, conservedFieldCount, 1>;
struct Primitive
{
    double rho = 0, T = 0, u = 0, v = 0, p = 0, c = 0, heatCapacityVolume = 0;
    std::array<double, 3> alpha{}, density{}, species{};
};
struct Settings
{
    bool reactions = true, conduction = true;
    bool openBoundaries = false;
    std::vector<int> fixedCells; // Stationary end-support cells selected at initialization.
    double thincBeta = 1.6;
    double pistonVelocity = 0; // Constant laboratory velocity of translating mesh/wall.
};

// Decode phase densities from PARTIAL masses and volume fractions. Species are
// conditional mass fractions within RDX, never fractions of the entire mixture.
Primitive decode(const Cell &U)
{
    if (!U.allFinite())
        throw std::runtime_error("Nonfinite state");
    Primitive q;
    double fractionSum = 0;
    for (int phase = 0; phase < 3; ++phase)
    {
        const double storedAlpha = U[AlphaRDX + phase], mass = U[AlphaRhoRDX + phase];
        // Partial density can underflow before its fraction (rho is ~1e-9).
        // An exactly massless, subnormal fraction has no thermodynamic state.
        // Do not change any conserved mass or energy, or hide finite violations.
        const bool underflowed = mass == 0 && storedAlpha >= 0 &&
                                 storedAlpha < std::numeric_limits<double>::min();
        const double alpha = underflowed ? 0 : storedAlpha;
        if (alpha < 0 || alpha > 1 + 1e-10 || mass < 0 || (alpha == 0) != (mass == 0))
        {
            std::ostringstream message;
            message << std::setprecision(17) << "Invalid phase fraction or partial density: phase="
                    << phase << ", alpha=" << alpha << ", partial density=" << mass;
            throw std::runtime_error(message.str());
        }
        q.alpha[phase] = alpha;
        q.density[phase] = alpha > 0 ? mass / alpha : 0;
        q.rho += mass;
        fractionSum += alpha;
        q.heatCapacityVolume += mass * multimaterialEOS::heatCapacity(phase);
    }
    if (std::abs(fractionSum - 1) > 1e-9 || !(q.rho > 0))
        throw std::runtime_error("Phase fractions must sum to one; air is not vacuum");
    const double rdxMass = U[AlphaRhoRDX];
    double speciesSum = 0;
    for (int k = 0; k < 3; ++k)
    {
        if (U[k] < 0 || U[k] > rdxMass * (1 + 1e-10))
            throw std::runtime_error("Invalid RDX species mass");
        q.species[k] = rdxMass > 0 ? U[k] / rdxMass : 0;
        speciesSum += U[k];
    }
    if (std::abs(speciesSum - rdxMass) > 1e-10 * std::max(rdxMass, 1e-30))
        throw std::runtime_error("RDX species masses do not sum to RDX partial density");
    q.u = U[MomentumX] / q.rho;
    q.v = U[MomentumY] / q.rho;
    double coldEnergy = 0;
    for (int phase = 0; phase < 3; ++phase)
        if (q.alpha[phase] > 0)
            coldEnergy += U[AlphaRhoRDX + phase] *
                          multimaterialEOS::referenceEnergy(phase, q.density[phase], q.species[0]);
    q.T =
        (U[TotalEnergy] - .5 * q.rho * (q.u * q.u + q.v * q.v) - coldEnergy) / q.heatCapacityVolume;
    if (!std::isfinite(q.T) || q.T <= 0)
        throw std::runtime_error("Nonpositive mixture temperature");
    double pressureT = 0, compressionDerivative = 0;
    for (int phase = 0; phase < 3; ++phase)
        if (q.alpha[phase] > 0)
        {
            const double rho = q.density[phase], a = q.alpha[phase], Y1 = q.species[0];
            q.p += a * multimaterialEOS::pressureValue(phase, rho, q.T, Y1);
            pressureT += a * multimaterialEOS::pressureTemperatureDerivative(phase, rho, q.T, Y1);
            compressionDerivative +=
                a * rho * multimaterialEOS::pressureDensityDerivative(phase, rho, q.T, Y1);
        }
    // Frozen alpha/composition, common-T adiabatic acoustic speed. This is not
    // Wood's pressure-equilibrium mixture sound speed.
    const double c2 = compressionDerivative / q.rho +
                      q.T * pressureT * pressureT / (q.rho * q.heatCapacityVolume);
    if (!std::isfinite(q.p) || !std::isfinite(c2) || c2 <= 0)
        throw std::runtime_error("Invalid mixture sound speed/pressure");
    q.c = std::sqrt(c2);
    return q;
}

Cell makeState(const std::array<double, 3> &alpha, const std::array<double, 3> &density, double T,
               double u = 0, double v = 0, const std::array<double, 3> &species = {1, 0, 0})
{
    Cell U = Cell::Zero();
    double rho = 0;
    for (int phase = 0; phase < 3; ++phase)
    {
        U[AlphaRDX + phase] = alpha[phase];
        if (alpha[phase] == 0)
            continue;
        if (!(density[phase] > 0))
            throw std::invalid_argument("Every present phase requires positive intrinsic density");
        const double mass = alpha[phase] * density[phase];
        U[AlphaRhoRDX + phase] = mass;
        rho += mass;
        U[TotalEnergy] +=
            mass * multimaterialEOS::internalEnergy(phase, density[phase], T, species[0]);
    }
    for (int k = 0; k < 3; ++k)
        U[k] = U[AlphaRhoRDX] * species[k];
    U[MomentumX] = rho * u;
    U[MomentumY] = rho * v;
    U[TotalEnergy] += .5 * rho * (u * u + v * v);
    decode(U);
    return U;
}
Cell pureState(int phase, double rho, double T = 300, double u = 0, double v = 0)
{
    std::array<double, 3> alpha{}, density{};
    alpha[phase] = 1;
    density[phase] = rho;
    return makeState(alpha, density, T, u, v);
}
void validate(const State &state, const Grid &grid)
{
    for (int cell = 0; cell < grid.size(); ++cell)
        try
        {
            decode(state.col(cell));
        }
        catch (const std::runtime_error &error)
        {
            throw std::runtime_error(std::string(error.what()) + " at cell (" +
                                     std::to_string(cell % grid.nx) + "," +
                                     std::to_string(cell / grid.nx) + ")");
        }
}

struct Evaluation
{
    State rhs;
    Conserved boundary = Conserved::Zero();
    double stepLimit = 0, reactionHeat = 0;
    long faceFallbacks = 0;
};

// Reconstruct all fields with common WENO weights in a single phase. At material
// interfaces, reconstruct/normalize all three THINC fractions together, then
// rebuild partial masses, RDX species and energy from each donor's intrinsic state.
std::pair<Cell, Cell> reconstruct(const std::array<Cell, 6> &cells, bool highOrder, double beta)
{
    std::array<std::array<double, 6>, stateFieldCount> stencil;
    bool sameFractions = true;
    for (int n = 0; n < 6; ++n)
    {
        for (int k = 0; k < stateFieldCount; ++k)
            stencil[k][n] = cells[n][k];
        for (int phase = 0; phase < 3; ++phase)
            if (cells[n][AlphaRDX + phase] != cells[2][AlphaRDX + phase])
                sameFractions = false;
    }
    const auto traces =
        WENO::reconstructSystem<stateFieldCount>(stencil, highOrder && sameFractions, MomentumX);
    Cell left, right;
    for (int k = 0; k < stateFieldCount; ++k)
    {
        left[k] = traces[k].left;
        right[k] = traces[k].right;
    }
    if (highOrder && !sameFractions)
        for (int side = 0; side < 2; ++side)
        {
            const int center = side + 2;
            const auto donor = decode(cells[center]);
            std::array<double, 3> alpha{};
            double sum = 0;
            for (int phase = 0; phase < 3; ++phase)
            {
                const int k = AlphaRDX + phase;
                const auto edge = THINC::reconstruct(cells[center - 1][k], cells[center][k],
                                                     cells[center + 1][k], beta);
                alpha[phase] = side == 0 ? edge.right : edge.left;
                if (donor.alpha[phase] == 0)
                    alpha[phase] = 0;
                sum += alpha[phase];
            }
            if (sum <= 0)
                throw std::runtime_error("Degenerate THINC composition");
            for (double &a : alpha)
                a /= sum;
            (side == 0 ? left : right) =
                makeState(alpha, donor.density, donor.T, donor.u, donor.v, donor.species);
        }
    if (sameFractions)
        for (int phase = 0; phase < 3; ++phase)
            left[AlphaRDX + phase] = right[AlphaRDX + phase] = cells[2][AlphaRDX + phase];
    decode(left);
    decode(right);
    return {left, right};
}

Evaluation evaluate(const State &state, const Grid &grid, const Settings &settings,
                    bool highOrder = true)
{
    Evaluation result;
    result.rhs = State::Zero(stateFieldCount, grid.size());
    std::vector<Primitive> primitive(grid.size());
    double inverseDt = 0;
    for (int cell = 0; cell < grid.size(); ++cell)
    {
        const auto q = decode(state.col(cell));
        primitive[cell] = q;
        inverseDt = std::max(inverseDt,
                             (std::abs(q.u - settings.pistonVelocity) + q.c) / grid.dx() + (std::abs(q.v) + q.c) / grid.dy() +
                                 2 * (settings.conduction ? params::k : 0) / q.heatCapacityVolume *
                                     (1 / (grid.dx() * grid.dx()) + 1 / (grid.dy() * grid.dy())));
    }
    auto sample = [&](int i, int j) -> Cell
    {
        if (i < 0 && !settings.openBoundaries) {
            const Cell interior = state.col(grid.index(std::clamp(-i-1,0,grid.nx-1),std::clamp(j,0,grid.ny-1)));
            const auto q = decode(interior);
            Cell ghost = interior;
            const double reflectedVelocity = 2*settings.pistonVelocity-q.u;
            ghost[MomentumX] = q.rho*reflectedVelocity;
            ghost[TotalEnergy] += .5*q.rho*(reflectedVelocity*reflectedVelocity-q.u*q.u);
            return ghost;
        }
        return state.col(grid.index(std::clamp(i, 0, grid.nx - 1), std::clamp(j, 0, grid.ny - 1)));
    };
    auto face = [&](int i, int j, bool alongX)
    {
        std::array<Cell, 6> cells;
        for (int n = 0; n < 6; ++n)
            cells[n] = sample(i + (alongX ? n - 3 : 0), j + (alongX ? 0 : n - 3));
        const bool interior = alongX ? i >= 3 && i <= grid.nx - 3 : j >= 3 && j <= grid.ny - 3;
        Cell left = cells[2], right = cells[3];
        if (highOrder && interior)
            try
            {
                auto traces = reconstruct(cells, true, settings.thincBeta);
                left = traces.first;
                right = traces.second;
            }
            catch (const std::runtime_error &)
            {
                ++result.faceFallbacks;
            }
        const auto qL = decode(left), qR = decode(right);
        const double uL = alongX ? qL.u : qL.v, uR = alongX ? qR.u : qR.v;
        const double meshVelocity = alongX ? settings.pistonVelocity : 0;
        const double speed = std::max(std::abs(uL-meshVelocity) + qL.c, std::abs(uR-meshVelocity) + qR.c);
        const double spacing = alongX ? grid.dx() : grid.dy();
        inverseDt = std::max(inverseDt, 2 * speed / spacing);
        Cell fluxL = (uL-meshVelocity) * left, fluxR = (uR-meshVelocity) * right;
        fluxL[alongX ? MomentumX : MomentumY] += qL.p;
        fluxR[alongX ? MomentumX : MomentumY] += qR.p;
        fluxL[TotalEnergy] += uL * qL.p;
        fluxR[TotalEnergy] += uR * qR.p;
        Cell flux = .5 * (fluxL + fluxR) - .5 * speed * (right - left);
        const bool pistonFace = alongX && i == 0 && !settings.openBoundaries;
        if (pistonFace) {
            // Impermeable, adiabatic slip wall. Rusanov reflected-state traction;
            // impose exact mechanical work p_wall * v_piston, not ghost energy mixing.
            const double wallPressure = flux[MomentumX];
            flux.setZero();
            flux[MomentumX] = wallPressure;
            flux[TotalEnergy] = wallPressure * settings.pistonVelocity;
        }
        if (settings.conduction && !pistonFace)
            flux[TotalEnergy] -= params::k * (decode(cells[3]).T - decode(cells[2]).T) / spacing;
        const bool hasLeft = alongX ? i > 0 : j > 0, hasRight = alongX ? i < grid.nx : j < grid.ny;
        const int l = grid.index(std::clamp(i - (alongX ? 1 : 0), 0, grid.nx - 1),
                                 std::clamp(j - (alongX ? 0 : 1), 0, grid.ny - 1));
        const int r = grid.index(std::clamp(i, 0, grid.nx - 1), std::clamp(j, 0, grid.ny - 1));
        if (hasLeft)
            result.rhs.col(l) -= flux / spacing;
        else
            result.boundary += flux.head<conservedFieldCount>() * (alongX ? grid.dy() : grid.dx());
        if (hasRight)
            result.rhs.col(r) += flux / spacing;
        else
            result.boundary -= flux.head<conservedFieldCount>() * (alongX ? grid.dy() : grid.dx());
        // D alpha_k / Dt = 0. Partial masses remain strictly conservative.
        const double velocity = pistonFace ? 0 : .5 * (uL + uR) - meshVelocity;
        for (int phase = 0; phase < 3; ++phase)
        {
            const int k = AlphaRDX + phase;
            if (hasLeft)
                result.rhs(k, l) += state(k, l) * velocity / spacing;
            if (hasRight)
                result.rhs(k, r) -= state(k, r) * velocity / spacing;
        }
    };
    for (int j = 0; j < grid.ny; ++j)
        for (int i = 0; i <= grid.nx; ++i)
            face(i, j, true);
    for (int j = 0; j <= grid.ny; ++j)
        for (int i = 0; i < grid.nx; ++i)
            face(i, j, false);
    result.stepLimit = .08 / std::max(inverseDt, 1e-30);
    if (settings.reactions)
        for (int cell = 0; cell < grid.size(); ++cell)
            if (state(AlphaRhoRDX, cell) > 0)
            {
                const auto &q = primitive[cell];
                const double r1 =
                    reactionRates::computeArrhenius(q.T, params::R, params::E1, params::Z1);
                const double r2 =
                    reactionRates::computeArrhenius(q.T, params::R, params::E2, params::Z2);
                const double rate1 = r1 * state(Species1Mass, cell),
                             rate2 = r2 * state(Species2Mass, cell);
                const double heat =
                    (params::rho_init / q.density[0]) *
                    (reactionRates::computeReactionHeat(params::a1, params::b1, q.T) * rate1 +
                     reactionRates::computeReactionHeat(params::a2, params::b2, q.T) * rate2);
                result.rhs(Species1Mass, cell) -= rate1;
                result.rhs(Species2Mass, cell) += rate1 - rate2;
                result.rhs(Species3Mass, cell) += rate2;
                result.rhs(TotalEnergy, cell) += heat;
                result.reactionHeat += heat * grid.dx() * grid.dy();
                result.stepLimit = std::min(result.stepLimit, .08 / std::max({r1, r2, 1e-30}));
                const double referenceChange =
                    rate1 * std::abs(multimaterialEOS::rdxReferenceEnergy(q.density[0], false) -
                                     multimaterialEOS::rdxReferenceEnergy(q.density[0], true));
                result.stepLimit = std::min(result.stepLimit,
                                            .05 * q.heatCapacityVolume * q.T /
                                                std::max(std::abs(heat) + referenceChange, 1e-30));
            }
    return result;
}
struct Step
{
    State state;
    Conserved boundary;
    double reactionHeat;
    long faceFallbacks;
};
Step RK3Step(const State &initial, const Grid &grid, const Settings &settings, double dt,
             bool highOrder, const Evaluation &r0)
{
    Conserved supportExchange = Conserved::Zero();
    auto advance = [&](const State &state, const Evaluation &rate, double weight) -> State
    {
        if (dt > rate.stepLimit * (1 + 1e-12))
            throw std::runtime_error("Stage CFL/source limit exceeded");
        State next = state + dt * rate.rhs;
        // The stationary support removes mechanical work; it does not heat the film.
        for (int cell : settings.fixedCells)
        {
            const double rho = next(AlphaRhoRDX,cell) + next(AlphaRhoPS,cell) + next(AlphaRhoPore,cell);
            if (rho <= 0) throw std::runtime_error("Invalid density at support");
            const double mx = next(MomentumX,cell), my = next(MomentumY,cell);
            const double kinetic = .5*(mx*mx+my*my)/rho;
            next(MomentumX,cell) = next(MomentumY,cell) = 0;
            next(TotalEnergy,cell) -= kinetic;
            const double volumeWeight = weight*grid.dx()*grid.dy();
            supportExchange[MomentumX] -= volumeWeight*mx;
            supportExchange[MomentumY] -= volumeWeight*my;
            supportExchange[TotalEnergy] -= volumeWeight*kinetic;
        }
        validate(next, grid);
        return next;
    };
    const State stage1 = advance(initial, r0, 1.0/6);
    const auto r1 = evaluate(stage1, grid, settings, highOrder);
    const State stage2 = .75 * initial + .25 * advance(stage1, r1, 1.0/6);
    validate(stage2, grid);
    const auto r2 = evaluate(stage2, grid, settings, highOrder);
    State finalState = initial / 3 + (2.0 / 3) * advance(stage2, r2, 2.0/3);
    validate(finalState, grid);
    return {std::move(finalState), dt * (r0.boundary / 6 + r1.boundary / 6 + 2 * r2.boundary / 3) + supportExchange,
            dt * (r0.reactionHeat / 6 + r1.reactionHeat / 6 + 2 * r2.reactionHeat / 3),
            r0.faceFallbacks + r1.faceFallbacks + r2.faceFallbacks};
}

#ifndef MECHANICS2DMULTIMATERIAL_NO_MAIN
int main(int argc, char **argv)
{
    if (argc < 11)
    {
        std::cerr << "Usage: " << argv[0]
                  << " Nx Ny Lx Ly dt output.e|csv steps stride piston_velocity radius [--density-map "
                     "x_y_rho_phase.csv] [--thinc-beta value] [--no-chemistry] [--no-conduction] [--rdx-velocity vx vy] [--open-boundaries] [--fixed-ps-ends width]\n";
        return 1;
    }
    try
    {
        const Grid grid{std::stoi(argv[1]), std::stoi(argv[2]), std::stod(argv[3]),
                        std::stod(argv[4])};
        const double interval = std::stod(argv[5]), pistonVelocity = std::stod(argv[9]),
                     radius = std::stod(argv[10]);
        const int steps = std::stoi(argv[7]), stride = std::stoi(argv[8]);
        const std::string outputPath = argv[6];
        std::string mapPath;
        Settings settings;
        double rdxVx = 0, rdxVy = 0, supportWidth = 0;
        for (int a = 11; a < argc; ++a)
        {
            const std::string option = argv[a];
            if (option == "--density-map" && a + 1 < argc)
                mapPath = argv[++a];
            else if (option == "--thinc-beta" && a + 1 < argc)
                settings.thincBeta = std::stod(argv[++a]);
            else if (option == "--rdx-velocity" && a + 2 < argc)
            {
                rdxVx = std::stod(argv[++a]);
                rdxVy = std::stod(argv[++a]);
            }
            else if (option == "--fixed-ps-ends" && a + 1 < argc)
                supportWidth = std::stod(argv[++a]);
            else if (option == "--open-boundaries")
                settings.openBoundaries = true;
            else if (option == "--no-chemistry")
                settings.reactions = false;
            else if (option == "--no-conduction")
                settings.conduction = false;
            else
                throw std::invalid_argument("Unknown/incomplete option: " + option);
        }
        for (double value : {grid.lx, grid.ly, interval, pistonVelocity, radius, settings.thincBeta, rdxVx, rdxVy, supportWidth})
            if (!std::isfinite(value))
                throw std::invalid_argument("Nonfinite input");
        if (grid.nx < 6 || grid.ny < 6 || grid.lx <= 0 || grid.ly <= 0 || interval <= 0 ||
            pistonVelocity < 0 || steps < 0 || stride < 1 || settings.thincBeta <= 0 ||
            settings.thincBeta > 10)
            throw std::invalid_argument("Invalid grid/time/THINC parameters");
        if ((settings.openBoundaries || supportWidth > 0) && pistonVelocity != 0)
            throw std::invalid_argument("Open boundaries/supports require zero piston velocity");
        if (supportWidth < 0 || supportWidth >= grid.lx/2)
            throw std::invalid_argument("Invalid support width");
        if (mapPath.empty() && (rdxVx != 0 || rdxVy != 0 || supportWidth > 0))
            throw std::invalid_argument("Impact velocity/supports require a phase density map");
        State state(stateFieldCount, grid.size());
        std::vector<int> phases(grid.size(), 0);
        Eigen::VectorXd densities = Eigen::VectorXd::Constant(grid.size(), params::rho_init);
        if (!mapPath.empty())
        {
            const auto map = densityMap2D::read(mapPath, true);
            for (int j = 0; j < grid.ny; ++j)
                for (int i = 0; i < grid.nx; ++i)
                {
                    const int cell = grid.index(i, j);
                    const double x = (i + .5) * grid.dx(), y = (j + .5) * grid.dy();
                    densities[cell] = map.at(x, y);
                    phases[cell] = static_cast<int>(map.phaseAt(x, y));
                    if (densities[cell] <= 0)
                        throw std::invalid_argument(
                            "All phases, including pore air, require positive CSV density");
                    state.col(cell) = pureState(phases[cell], densities[cell], 300,
                                               phases[cell] == 0 ? rdxVx : 0,
                                               phases[cell] == 0 ? rdxVy : 0);
                    if (phases[cell] == 1 && (x < supportWidth || x > grid.lx-supportWidth))
                        settings.fixedCells.push_back(cell);
                }
        }
        else
        {
            if (radius <= 0 || radius >= std::min(grid.lx / 4, grid.ly / 2))
                throw std::invalid_argument("Invalid circle radius");
            const auto geometry = levelSet2D::geometry(
                grid, levelSet2D::circle(grid, grid.lx / 4, grid.ly / 2, radius));
            for (int cell = 0; cell < grid.size(); ++cell)
            {
                const double solid = geometry.fraction[cell];
                state.col(cell) = makeState({solid, 0, 1 - solid},
                                            {params::rho_init, multimaterialEOS::rhoPS,
                                             multimaterialEOS::defaultAirDensity},
                                            300);
            }
        }
        if (supportWidth > 0 && settings.fixedCells.empty())
            throw std::invalid_argument("No PS cells found in end-support strips");
        validate(state, grid);
        settings.pistonVelocity = pistonVelocity;
        if (settings.openBoundaries) std::cout << "Stationary mesh, open boundaries; support cells " << settings.fixedCells.size() << "\n";
        else std::cout << "Piston velocity " << pistonVelocity << " nm/ns; translating mesh, impermeable left wall\n";
        const std::vector<std::string> names = {"Y1",
                                                "Y2",
                                                "Y3",
                                                "T",
                                                "rho",
                                                "vx",
                                                "vy",
                                                "rhoE",
                                                "p",
                                                "alpha_rdx",
                                                "alpha_ps",
                                                "alpha_pore",
                                                "alpha_rho_rdx",
                                                "alpha_rho_ps",
                                                "alpha_rho_pore",
                                                "rho_rdx",
                                                "rho_ps",
                                                "rho_pore", "disp_x", "disp_y"};
        std::ofstream csv;
        std::unique_ptr<ExodusOutput> exodus;
        const auto extension = std::filesystem::path(outputPath).extension();
        if (extension == ".csv")
        {
            csv.exceptions(std::ios::badbit | std::ios::failbit);
            csv.open(outputPath);
            csv << std::setprecision(17) << "step,time_ns,X,Y";
            for (const auto &name : names)
                csv << ',' << name;
            csv << '\n';
        }
        else if (extension == ".e" || extension == ".exo" || extension == ".exodus")
            exodus =
                std::make_unique<ExodusOutput>(outputPath, grid.nx, grid.ny, grid.dx(), grid.dy(),
                                               names, .5 * grid.dx(), .5 * grid.dy());
        else
            throw std::invalid_argument("Output must be .csv or .e/.exo/.exodus");
        const double volume = grid.dx() * grid.dy();
        const Conserved initialTotals =
            state.topRows<conservedFieldCount>().rowwise().sum() * volume;
        Conserved boundary = Conserved::Zero();
        double reactionHeat = 0;
        long substeps = 0, retries = 0, fallbackSteps = 0;
        std::ofstream diagnostics(outputPath + ".diagnostics.csv");
        diagnostics.exceptions(std::ios::failbit | std::ios::badbit);
        diagnostics
            << std::setprecision(17)
            << "time_ns,rdx_mass_error,ps_mass_error,air_mass_error,mx_error,my_error,energy_error,"
               "Tmin,Tmax,max_fraction_sum_error,internal_steps,retries,low_order_steps\n";
        auto save = [&](int step, double time)
        {
            std::vector<Eigen::VectorXd> fields(names.size(), Eigen::VectorXd(grid.size()));
            double minT = 1e300, maxT = 0, maxFractionError = 0;
            for (int cell = 0; cell < grid.size(); ++cell)
            {
                const auto q = decode(state.col(cell));
                for (int k = 0; k < 3; ++k)
                    fields[k][cell] = q.species[k];
                fields[3][cell] = q.T;
                fields[4][cell] = q.rho;
                fields[5][cell] = q.u;
                fields[6][cell] = q.v;
                fields[7][cell] = state(TotalEnergy, cell);
                fields[8][cell] = q.p;
                for (int phase = 0; phase < 3; ++phase)
                {
                    fields[9 + phase][cell] = q.alpha[phase];
                    fields[12 + phase][cell] = state(AlphaRhoRDX + phase, cell);
                    fields[15 + phase][cell] = q.density[phase];
                }
                fields[18][cell] = settings.pistonVelocity*time;
                fields[19][cell] = 0;
                minT = std::min(minT, q.T);
                maxT = std::max(maxT, q.T);
                maxFractionError =
                    std::max(maxFractionError, std::abs(q.alpha[0] + q.alpha[1] + q.alpha[2] - 1));
            }
            if (exodus)
                exodus->write(time, fields);
            else
                for (int j = 0; j < grid.ny; ++j)
                    for (int i = 0; i < grid.nx; ++i)
                    {
                        const int cell = grid.index(i, j);
                        csv << step << ',' << time << ',' << (i + .5) * grid.dx() + settings.pistonVelocity*time << ','
                            << (j + .5) * grid.dy();
                        for (const auto &f : fields)
                            csv << ',' << f[cell];
                        csv << '\n';
                    }
            Conserved error = state.topRows<conservedFieldCount>().rowwise().sum() * volume -
                              initialTotals - boundary;
            error[TotalEnergy] -= reactionHeat;
            diagnostics << time;
            for (int k = AlphaRhoRDX; k <= TotalEnergy; ++k)
                diagnostics << ',' << error[k];
            diagnostics << ',' << minT << ',' << maxT << ',' << maxFractionError << ',' << substeps
                        << ',' << retries << ',' << fallbackSteps << '\n';
            std::cout << "time " << time << " ns, T=[" << minT << ',' << maxT << "], energy error "
                      << error[TotalEnergy] << '\n';
        };
        double time = 0;
        save(0, time);
        for (int step = 1; step <= steps; ++step)
        {
            const double target = step * interval;
            while (time < target)
            {
                const auto rate = evaluate(state, grid, settings);
                double dt = std::min(target - time, .9 * rate.stepLimit);
                bool accepted = false;
                std::string lastError;
                for (int attempt = 0; attempt < 24 && !accepted; ++attempt)
                {
                    if (time + dt == time)
                        break;
                    for (bool highOrder : {true, false})
                        try
                        {
                            auto next =
                                RK3Step(state, grid, settings, dt, highOrder,
                                        highOrder ? rate : evaluate(state, grid, settings, false));
                            state = std::move(next.state);
                            boundary += next.boundary;
                            reactionHeat += next.reactionHeat;
                            time += dt;
                            ++substeps;
                            if (!highOrder)
                                ++fallbackSteps;
                            accepted = true;
                            break;
                        }
                        catch (const std::runtime_error &error)
                        {
                            lastError = error.what();
                        }
                    if (!accepted)
                    {
                        dt *= .5;
                        ++retries;
                    }
                }
                if (!accepted)
                    throw std::runtime_error("Cannot advance at time " + std::to_string(time) +
                                             ": " + lastError);
            }
            if (step % stride == 0 || step == steps)
                save(step, time);
        }
        if (exodus)
            exodus->close();
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#endif
