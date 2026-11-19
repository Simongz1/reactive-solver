// Three phases with a common velocity and temperature, and volume-weighted pressure.
// Deviatoric stress and plasticity are not included. Pores contain ideal-gas air.
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
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
#ifdef _OPENMP
#include <omp.h>
#endif

using Grid = levelSet2D::Grid;

// Runs body(index) for every index in [0, count). Built with OpenMP, iterations
// are shared among threads; each iteration must only write its own outputs.
// An exception from any iteration is rethrown after the loop, taking the lowest
// failing index so the error (and the retry it triggers) matches a serial run.
template <class Body>
void parallelFor(int count, const Body &body)
{
#ifdef _OPENMP
    std::exception_ptr error;
    int errorIndex = count;
#pragma omp parallel for schedule(static)
    for (int index = 0; index < count; ++index)
    {
        try
        {
            body(index);
        }
        catch (...)
        {
#pragma omp critical(parallelForError)
            if (index < errorIndex)
            {
                errorIndex = index;
                error = std::current_exception();
            }
        }
    }
    if (error)
        std::rethrow_exception(error);
#else
    for (int index = 0; index < count; ++index)
        body(index);
#endif
}
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
    // A failed high-order step is retried at dt/2, ..., dt/2^N before falling
    // back to local first-order fluxes around the cells that failed.
    int highOrderHalvings = 3;
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
    parallelFor(grid.size(), [&](int cell)
    {
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
    });
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

// lowOrderCells (optional, one flag per cell): every face touching a flagged
// cell uses first-order (cell-average) Rusanov states; all others stay WENO/THINC.
Evaluation evaluate(const State &state, const Grid &grid, const Settings &settings,
                    bool highOrder = true, const std::vector<char> *lowOrderCells = nullptr)
{
    // Parallel structure: (1) decode every cell, (2) compute every face flux
    // independently, (3) gather fluxes into each cell's rhs, (4) local chemistry.
    // No step writes to memory owned by another iteration, and every reduction
    // (time-step limit, boundary fluxes, heat release) is done afterwards in the
    // serial order, so threaded and serial runs give bit-identical results.
    Evaluation result;
    result.rhs.resize(stateFieldCount, grid.size()); // zeroed per cell in the gather
    std::vector<Primitive> primitive(grid.size());
    std::vector<double> cellInverseDt(grid.size());
    parallelFor(grid.size(), [&](int cell)
    {
        const auto q = decode(state.col(cell));
        primitive[cell] = q;
        cellInverseDt[cell] =
            (std::abs(q.u - settings.pistonVelocity) + q.c) / grid.dx() + (std::abs(q.v) + q.c) / grid.dy() +
            2 * (settings.conduction ? params::k : 0) / q.heatCapacityVolume *
                (1 / (grid.dx() * grid.dx()) + 1 / (grid.dy() * grid.dy()));
    });
    double inverseDt = 0;
    for (double value : cellInverseDt)
        inverseDt = std::max(inverseDt, value);
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
    // A face flux depends only on the six cells across it, so faces are computed
    // independently and stored; nothing is accumulated inside this function.
    // No default member values: the face arrays are allocated without a serial
    // initialization pass, and every field is written by face() in parallel.
    struct FaceFlux
    {
        Cell flux;
        double velocity, inverseDt;
        bool fallback;
    };
    auto face = [&](int i, int j, bool alongX) -> FaceFlux
    {
        FaceFlux result;
        result.fallback = false;
        std::array<Cell, 6> cells;
        for (int n = 0; n < 6; ++n)
            cells[n] = sample(i + (alongX ? n - 3 : 0), j + (alongX ? 0 : n - 3));
        const bool interior = alongX ? i >= 3 && i <= grid.nx - 3 : j >= 3 && j <= grid.ny - 3;
        Cell left = cells[2], right = cells[3];
        // Interior faces always have both neighbouring cells inside the grid.
        const bool locallyLowOrder =
            lowOrderCells && interior &&
            ((*lowOrderCells)[alongX ? grid.index(i - 1, j) : grid.index(i, j - 1)] ||
             (*lowOrderCells)[grid.index(i, j)]);
        if (highOrder && interior && !locallyLowOrder)
            try
            {
                auto traces = reconstruct(cells, true, settings.thincBeta);
                left = traces.first;
                right = traces.second;
            }
            catch (const std::runtime_error &)
            {
                result.fallback = true;
            }
        const auto qL = decode(left), qR = decode(right);
        const double uL = alongX ? qL.u : qL.v, uR = alongX ? qR.u : qR.v;
        const double meshVelocity = alongX ? settings.pistonVelocity : 0;
        const double speed = std::max(std::abs(uL-meshVelocity) + qL.c, std::abs(uR-meshVelocity) + qR.c);
        const double spacing = alongX ? grid.dx() : grid.dy();
        result.inverseDt = 2 * speed / spacing;
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
        result.flux = flux;
        // D alpha_k / Dt = 0. Partial masses remain strictly conservative.
        result.velocity = pistonFace ? 0 : .5 * (uL + uR) - meshVelocity;
        return result;
    };
    // x-face (i, j) sits between cells (i-1, j) and (i, j), for i = 0..nx;
    // y-face (i, j) sits between cells (i, j-1) and (i, j), for j = 0..ny.
    const int xFacesPerRow = grid.nx + 1;
    const int xFaceCount = xFacesPerRow * grid.ny, yFaceCount = grid.nx * (grid.ny + 1);
    const std::unique_ptr<FaceFlux[]> xFaces(new FaceFlux[xFaceCount]),
        yFaces(new FaceFlux[yFaceCount]);
    parallelFor(xFaceCount, [&](int f)
                { xFaces[f] = face(f % xFacesPerRow, f / xFacesPerRow, true); });
    parallelFor(yFaceCount, [&](int f)
                { yFaces[f] = face(f % grid.nx, f / grid.nx, false); });
    // Gather: each cell adds its own four faces, in the order the original
    // face-by-face scatter visited them (x faces i, i+1, then y faces j, j+1).
    parallelFor(grid.size(), [&](int cell)
    {
        const int i = cell % grid.nx, j = cell / grid.nx;
        auto rhs = result.rhs.col(cell);
        rhs.setZero();
        auto inflow = [&](const FaceFlux &f, double spacing) // cell is right of the face
        {
            rhs += f.flux / spacing;
            for (int k = AlphaRDX; k <= AlphaPore; ++k)
                rhs[k] -= state(k, cell) * f.velocity / spacing;
        };
        auto outflow = [&](const FaceFlux &f, double spacing) // cell is left of the face
        {
            rhs -= f.flux / spacing;
            for (int k = AlphaRDX; k <= AlphaPore; ++k)
                rhs[k] += state(k, cell) * f.velocity / spacing;
        };
        inflow(xFaces[j * xFacesPerRow + i], grid.dx());
        outflow(xFaces[j * xFacesPerRow + i + 1], grid.dx());
        inflow(yFaces[j * grid.nx + i], grid.dy());
        outflow(yFaces[(j + 1) * grid.nx + i], grid.dy());
    });
    // Serial reductions over faces, in the original visiting order.
    auto reduceFace = [&](const FaceFlux &f)
    {
        inverseDt = std::max(inverseDt, f.inverseDt);
        result.faceFallbacks += f.fallback;
    };
    for (int j = 0; j < grid.ny; ++j)
    {
        for (int i = 0; i <= grid.nx; ++i)
            reduceFace(xFaces[j * xFacesPerRow + i]);
        result.boundary += xFaces[j * xFacesPerRow].flux.head<conservedFieldCount>() * grid.dy();
        result.boundary -=
            xFaces[j * xFacesPerRow + grid.nx].flux.head<conservedFieldCount>() * grid.dy();
    }
    for (int f = 0; f < yFaceCount; ++f)
        reduceFace(yFaces[f]);
    for (int i = 0; i < grid.nx; ++i)
        result.boundary += yFaces[i].flux.head<conservedFieldCount>() * grid.dx();
    for (int i = 0; i < grid.nx; ++i)
        result.boundary -= yFaces[grid.ny * grid.nx + i].flux.head<conservedFieldCount>() * grid.dx();
    result.stepLimit = .08 / std::max(inverseDt, 1e-30);
    if (settings.reactions)
    {
        // Chemistry is local to each cell; heat release and time-step limits are
        // stored per cell and reduced afterwards in cell order.
        std::vector<double> cellHeat(grid.size(), 0), cellLimit(grid.size(), 0);
        parallelFor(grid.size(), [&](int cell)
        {
            if (!(state(AlphaRhoRDX, cell) > 0))
                return;
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
            cellHeat[cell] = heat;
            const double referenceChange =
                rate1 * std::abs(multimaterialEOS::rdxReferenceEnergy(q.density[0], false) -
                                 multimaterialEOS::rdxReferenceEnergy(q.density[0], true));
            cellLimit[cell] = std::min(.08 / std::max({r1, r2, 1e-30}),
                                       .05 * q.heatCapacityVolume * q.T /
                                           std::max(std::abs(heat) + referenceChange, 1e-30));
        });
        for (int cell = 0; cell < grid.size(); ++cell)
            if (state(AlphaRhoRDX, cell) > 0)
            {
                result.reactionHeat += cellHeat[cell] * grid.dx() * grid.dy();
                result.stepLimit = std::min(result.stepLimit, cellLimit[cell]);
            }
    }
    return result;
}
// Elementwise state arithmetic, one cell (column) per iteration across threads.
template <class Combine>
State combineStates(int cells, const Combine &combine)
{
    State result(stateFieldCount, cells);
    parallelFor(cells, [&](int cell) { result.col(cell) = combine(cell); });
    return result;
}
struct Step
{
    State state;
    Conserved boundary;
    double reactionHeat;
    long faceFallbacks;
};
// Cells whose state cannot be decoded (instead of throwing at the first one).
std::vector<int> invalidCells(const State &state, const Grid &grid)
{
    std::vector<char> bad(grid.size(), 0);
    parallelFor(grid.size(), [&](int cell)
    {
        try
        {
            decode(state.col(cell));
        }
        catch (const std::runtime_error &)
        {
            bad[cell] = 1;
        }
    });
    std::vector<int> cells;
    for (int cell = 0; cell < grid.size(); ++cell)
        if (bad[cell])
            cells.push_back(cell);
    return cells;
}

// lowOrderCells (optional, zeroed by the caller) enables a-posteriori local
// limiting: when a stage leaves invalid cells, those cells are flagged, their
// faces are recomputed with first-order Rusanov states, and the stage is redone.
// A cell that fails again while flagged also flags its 3x3 neighbourhood. Fluxes
// are still shared by both neighbours of a face, so the update stays conservative.
// Without lowOrderCells, any invalid cell throws (the caller then cuts dt).
Step RK3Step(const State &initial, const Grid &grid, const Settings &settings, double dt,
             bool highOrder, const Evaluation &r0, std::vector<char> *lowOrderCells = nullptr)
{
    constexpr int maxLocalPasses = 4;
    const bool local = highOrder && lowOrderCells;
    Conserved supportExchange = Conserved::Zero();
    // Forward-Euler stage; the support exchange is only kept once the stage is accepted.
    auto euler = [&](const State &state, const Evaluation &rate, double weight,
                     Conserved &exchange) -> State
    {
        if (dt > rate.stepLimit * (1 + 1e-12))
            throw std::runtime_error("Stage CFL/source limit exceeded");
        State next = combineStates(grid.size(), [&](int cell) -> Cell
                                   { return state.col(cell) + dt * rate.rhs.col(cell); });
        exchange.setZero();
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
            exchange[MomentumX] -= volumeWeight*mx;
            exchange[MomentumY] -= volumeWeight*my;
            exchange[TotalEnergy] -= volumeWeight*kinetic;
        }
        return next;
    };
    // rate may be redirected to `storage`, a locally first-order re-evaluation of `state`.
    auto advance = [&](const State &state, const Evaluation *&rate, Evaluation &storage,
                       double weight) -> State
    {
        for (int pass = 0;; ++pass)
        {
            Conserved exchange;
            State next = euler(state, *rate, weight, exchange);
            if (!local)
                validate(next, grid);
            else
            {
                const auto bad = invalidCells(next, grid);
                if (!bad.empty())
                {
                    bool widened = false;
                    for (int cell : bad)
                    {
                        if (!(*lowOrderCells)[cell])
                        {
                            (*lowOrderCells)[cell] = 1;
                            widened = true;
                            continue;
                        }
                        const int i = cell % grid.nx, j = cell / grid.nx;
                        for (int dj = -1; dj <= 1; ++dj)
                            for (int di = -1; di <= 1; ++di)
                            {
                                const int ni = i + di, nj = j + dj;
                                if (ni < 0 || nj < 0 || ni >= grid.nx || nj >= grid.ny)
                                    continue;
                                char &flag = (*lowOrderCells)[grid.index(ni, nj)];
                                widened |= !flag;
                                flag = 1;
                            }
                    }
                    if (!widened || pass == maxLocalPasses)
                        validate(next, grid); // throws with the failing cell; caller cuts dt
                    storage = evaluate(state, grid, settings, true, lowOrderCells);
                    rate = &storage;
                    continue;
                }
            }
            supportExchange += exchange;
            return next;
        }
    };
    Evaluation e0Local, e1 = {}, e2 = {};
    const Evaluation *e0 = &r0, *p1 = &e1, *p2 = &e2;
    const State stage1 = advance(initial, e0, e0Local, 1.0/6);
    e1 = evaluate(stage1, grid, settings, highOrder, lowOrderCells);
    const State advanced1 = advance(stage1, p1, e1, 1.0/6);
    const State stage2 = combineStates(grid.size(), [&](int cell) -> Cell
                                       { return .75 * initial.col(cell) + .25 * advanced1.col(cell); });
    validate(stage2, grid);
    e2 = evaluate(stage2, grid, settings, highOrder, lowOrderCells);
    const State advanced2 = advance(stage2, p2, e2, 2.0/3);
    State finalState = combineStates(grid.size(), [&](int cell) -> Cell
                                     { return initial.col(cell) / 3 + (2.0 / 3) * advanced2.col(cell); });
    validate(finalState, grid);
    return {std::move(finalState), dt * (e0->boundary / 6 + e1.boundary / 6 + 2 * e2.boundary / 3) + supportExchange,
            dt * (e0->reactionHeat / 6 + e1.reactionHeat / 6 + 2 * e2.reactionHeat / 3),
            e0->faceFallbacks + e1.faceFallbacks + e2.faceFallbacks};
}

#ifndef MECHANICS2DMULTIMATERIAL_NO_MAIN
int main(int argc, char **argv)
{
    if (argc < 11)
    {
        std::cerr << "Usage: " << argv[0]
                  << " Nx Ny Lx Ly dt output.e|csv steps stride piston_velocity radius [--density-map "
                     "x_y_rho_phase.csv] [--thinc-beta value] [--no-chemistry] [--no-conduction] [--rdx-velocity vx vy] [--open-boundaries] [--fixed-ps-ends width] [--high-order-halvings n]\n";
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
            else if (option == "--high-order-halvings" && a + 1 < argc)
                settings.highOrderHalvings = std::stoi(argv[++a]);
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
        if (settings.highOrderHalvings < 0 || settings.highOrderHalvings > 20)
            throw std::invalid_argument("--high-order-halvings must be between 0 and 20");
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
#ifdef _OPENMP
        std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
#endif
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
        long substeps = 0, retries = 0, fallbackSteps = 0, lowOrderCellUpdates = 0,
             globalFallbackSteps = 0;
        std::vector<char> lowOrderCells(grid.size(), 0);
        std::ofstream diagnostics(outputPath + ".diagnostics.csv");
        diagnostics.exceptions(std::ios::failbit | std::ios::badbit);
        diagnostics
            << std::setprecision(17)
            << "time_ns,rdx_mass_error,ps_mass_error,air_mass_error,mx_error,my_error,energy_error,"
               "Tmin,Tmax,max_fraction_sum_error,internal_steps,retries,low_order_steps,"
               "low_order_cells,global_low_order_steps\n";
        auto save = [&](int step, double time)
        {
            std::vector<Eigen::VectorXd> fields(names.size(), Eigen::VectorXd(grid.size()));
            double minT = 1e300, maxT = 0, maxFractionError = 0;
            parallelFor(grid.size(), [&](int cell)
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
            });
            for (int cell = 0; cell < grid.size(); ++cell)
            {
                minT = std::min(minT, fields[3][cell]);
                maxT = std::max(maxT, fields[3][cell]);
                maxFractionError = std::max(
                    maxFractionError,
                    std::abs(fields[9][cell] + fields[10][cell] + fields[11][cell] - 1));
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
                        << ',' << retries << ',' << fallbackSteps << ',' << lowOrderCellUpdates << ','
                        << globalFallbackSteps << '\n';
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
                const double fullDt = std::min(target - time, .9 * rate.stepLimit);
                bool accepted = false;
                std::string lastError;
                // mode 0: WENO/THINC everywhere; 1: local first order on flagged cells;
                // 2: first order everywhere (last resort).
                auto attempt = [&](double dt, int mode) -> bool
                {
                    if (time + dt == time)
                        return false;
                    if (mode == 1)
                        std::fill(lowOrderCells.begin(), lowOrderCells.end(), 0);
                    try
                    {
                        auto next = RK3Step(state, grid, settings, dt, mode != 2,
                                            mode == 2 ? evaluate(state, grid, settings, false) : rate,
                                            mode == 1 ? &lowOrderCells : nullptr);
                        state = std::move(next.state);
                        boundary += next.boundary;
                        reactionHeat += next.reactionHeat;
                        time += dt;
                        ++substeps;
                        if (mode == 1)
                        {
                            const long flagged =
                                std::count(lowOrderCells.begin(), lowOrderCells.end(), 1);
                            fallbackSteps += flagged > 0;
                            lowOrderCellUpdates += flagged;
                        }
                        if (mode == 2)
                            ++globalFallbackSteps;
                        return true;
                    }
                    catch (const std::runtime_error &error)
                    {
                        lastError = error.what();
                        ++retries;
                        return false;
                    }
                };
                // 1) Keep high order everywhere and cut dt: fullDt, fullDt/2, ...
                double dt = fullDt;
                for (int cut = 0; cut <= settings.highOrderHalvings && !accepted; ++cut, dt *= .5)
                    accepted = attempt(dt, 0);
                // 2) Then local first-order Rusanov only around the failing cells,
                //    again starting from the full step and cutting dt if needed.
                dt = fullDt;
                for (int cut = 0; cut < 24 && !accepted; ++cut, dt *= .5)
                    accepted = attempt(dt, 1);
                // 3) Last resort: the old global first-order step.
                dt = fullDt;
                for (int cut = 0; cut < 24 && !accepted; ++cut, dt *= .5)
                    accepted = attempt(dt, 2);
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
