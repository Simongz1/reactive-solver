// 2D multimaterial finite-volume solver for impact problems (LIPIT).
//
// Three phases (RDX, PS, air) share one velocity and one temperature. Each solid
// phase (RDX, PS) carries its own deviatoric stress (s11, s22, s12) and its own
// accumulated plastic strain. Air has no strength.
//
// Units: nm, ns, GPa; density is kg/m^3 * 1e-9.
//
// Equations (per phase k, mixture quantities without index):
//   d(alpha_k rho_k)/dt     + div(alpha_k rho_k v)     = 0
//   d(rho v)/dt             + div(rho v v - sigma)      = 0,   sigma = -p I + S
//   d(rho E)/dt             + div(rho E v - sigma v)    = div(k grad T) + reaction heat
//   d(alpha_k rho_k s_k)/dt + div(alpha_k rho_k s_k v)  = alpha_k rho_k (2 G_k dev(D) + W s_k - s_k W)
//   d(alpha_k rho_k ep_k)/dt+ div(alpha_k rho_k ep_k v) = 0    (plastic increments added after each step)
//   d(alpha_k)/dt           + div(alpha_k v)            = alpha_k div(v)
// with D = sym(grad v), W = skew(grad v), S = sum_k alpha_k s_k and
// E = internal + kinetic + elastic energy (s_k:s_k / 4 G_k per unit phase volume).
//
// Plane strain: D33 = 0 but s33 = -(s11 + s22) because s is trace free.
//
// Plasticity: after each RK3 step, every solid phase whose von Mises stress exceeds
// its yield stress is returned radially to the yield surface. Since the total
// energy is not changed, the elastic energy removed becomes heat.

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "modelParams.h"
#include "multimaterialEOS.h"
#include "computeReactionRates.h"
#include "densityMap2D.h"
#include "plasticityRDX.h"
#include "WENO.h"
#include "THINC.h"
#include "ExodusOutput.h"
#ifdef _OPENMP
#include <omp.h>
#endif

//field indices in the state vector
enum Field
{
    //conserved fields
    RHOY1 = 0, RHOY2 = 1, RHOY3 = 2,          //RDX species partial densities
    MASS_RDX = 3, MASS_PS = 4, MASS_AIR = 5,  //phase partial densities alpha_k rho_k
    MOMX = 6, MOMY = 7,                       //mixture momentum
    ENERGY = 8,                               //total energy rho E
    S11_RDX = 9, S22_RDX = 10, S12_RDX = 11,  //alpha rho s for RDX
    S11_PS = 12, S22_PS = 13, S12_PS = 14,    //alpha rho s for PS
    EP_RDX = 15, EP_PS = 16,                  //alpha rho ep
    //volume fractions (not conserved)
    ALPHA_RDX = 17, ALPHA_PS = 18, ALPHA_AIR = 19
};
constexpr int nFields = 20;

//phases: only RDX and PS are solids. Per-phase fields are addressed as
//MASS_RDX + phase, ALPHA_RDX + phase, S11_RDX + 3 * phase, EP_RDX + phase.
enum Phase { RDX = 0, PS = 1, AIR = 2 };
const int nSolids = 2;

//state[field][cell], as in the 1D solvers
using State = std::vector<Eigen::VectorXd>;
//all fields of a single cell
using Vars = Eigen::Matrix<double, nFields, 1>;

//grid
int nx, ny;
double Lx, Ly, dx, dy;
int index(int i, int j) { return j * nx + i; }

//settings
double thincBeta = 1.6;
double cfl = 0.08;
bool reactions = true;
bool conduction = true;
//boundary of each side: true = slip wall, false = open (zero gradient)
bool wallLeft = true, wallRight = true, wallBottom = true, wallTop = true;
//piston speed. The solver works in the piston frame: the left side is a wall and
//the material enters from the right at -pistonVelocity. Output velocities and
//disp_x are shifted back to the laboratory frame.
double pistonVelocity = 0;
int highOrderHalvings = 3;
//phase fractions below this are Rusanov leakage, removed after every step
//(see removeTracePhases); 0 disables the cleanup
double traceFraction = 1e-8;
std::vector<int> supportCells; //PS cells held fixed (end supports)

//primitive variables of one cell
struct Primitive
{
    double rho = 0, u = 0, v = 0, T = 0, p = 0, c = 0;
    double heatCapacity = 0;                    //sum of alpha_k rho_k cv_k
    double alpha[3] = {0, 0, 0};
    double density[3] = {0, 0, 0};              //intrinsic phase densities
    double Y[3] = {0, 0, 0};                    //RDX species mass fractions
    double s11[2] = {0, 0}, s22[2] = {0, 0}, s12[2] = {0, 0}; //phase deviatoric stress
    double ep[2] = {0, 0};                      //phase plastic strain
    double S11 = 0, S22 = 0, S12 = 0;           //mixture deviatoric stress
};

double shearModulus(int phase)
{
    if (phase == RDX) return params::G_rdx;
    if (phase == PS) return params::G_ps;
    return 0.0;
}

double yieldStress(int phase, double ep, double rate, double T)
{
    if (phase == RDX) return std::max(plasticityRDX::yield(ep, rate, T), 0.0);
    return params::Y_ps;
}

//s:s for a plane-strain deviator (s33 = -(s11 + s22), s12 = s21)
double doubleContraction(double s11, double s22, double s12)
{
    const double s33 = -(s11 + s22);
    return s11 * s11 + s22 * s22 + s33 * s33 + 2.0 * s12 * s12;
}

double vonMises(double s11, double s22, double s12)
{
    return std::sqrt(1.5 * doubleContraction(s11, s22, s12));
}

Vars getCell(const State &state, int cell)
{
    Vars U;
    for (int k = 0; k < nFields; ++k) U[k] = state[k][cell];
    return U;
}

//conserved state -> primitive variables. Returns false if the state is not physical;
//reason (optional) then says which check failed.
bool computePrimitive(const Vars &U, Primitive &q, std::string *reason = nullptr)
{
    auto fail = [&](const char *message)
    {
        if (reason) *reason = message;
        return false;
    };
    q = Primitive();
    if (!U.allFinite()) return fail("non-finite value");

    //phase fractions and densities
    double fractionSum = 0;
    for (int phase = 0; phase < 3; ++phase)
    {
        double alpha = U[ALPHA_RDX + phase];
        const double mass = U[MASS_RDX + phase];
        //a phase whose mass has underflowed to zero can leave a subnormal fraction behind
        if (mass == 0 && alpha >= 0 && alpha < std::numeric_limits<double>::min()) alpha = 0;
        if (alpha < 0 || alpha > 1 + 1e-10 || mass < 0 || (alpha == 0) != (mass == 0))
            return fail("invalid volume fraction or partial density");

        q.alpha[phase] = alpha;
        q.density[phase] = alpha > 0 ? mass / alpha : 0;
        q.rho += mass;
        q.heatCapacity += mass * multimaterialEOS::heatCapacity(phase);
        fractionSum += alpha;
    }
    if (std::abs(fractionSum - 1) > 1e-9 || !(q.rho > 0)) return fail("volume fractions do not sum to one");

    //RDX species, as fractions of the RDX mass
    const double rdxMass = U[MASS_RDX];
    double speciesSum = 0;
    for (int k = 0; k < 3; ++k)
    {
        //high-order transport can leave round-off negatives in species that are ~0;
        //removeTracePhases clips them after every step
        if (U[RHOY1 + k] < -1e-10 * rdxMass || U[RHOY1 + k] > rdxMass * (1 + 1e-10)) return fail("invalid species mass");
        q.Y[k] = rdxMass > 0 ? U[RHOY1 + k] / rdxMass : 0;
        speciesSum += U[RHOY1 + k];
    }
    if (std::abs(speciesSum - rdxMass) > 1e-10 * std::max(rdxMass, 1e-30)) return fail("species do not sum to the RDX mass");

    q.u = U[MOMX] / q.rho;
    q.v = U[MOMY] / q.rho;

    //deviatoric stress and plastic strain of the solid phases
    double elasticEnergy = 0, shearStiffness = 0;
    for (int phase = 0; phase < nSolids; ++phase)
    {
        const double mass = U[MASS_RDX + phase];
        if (mass > 0)
        {
            q.s11[phase] = U[S11_RDX + 3 * phase] / mass;
            q.s22[phase] = U[S22_RDX + 3 * phase] / mass;
            q.s12[phase] = U[S12_RDX + 3 * phase] / mass;
            q.ep[phase] = U[EP_RDX + phase] / mass;
        }
        const double G = shearModulus(phase);
        if (G > 0)
            elasticEnergy += q.alpha[phase] * doubleContraction(q.s11[phase], q.s22[phase], q.s12[phase]) / (4.0 * G);
        shearStiffness += q.alpha[phase] * G;
        q.S11 += q.alpha[phase] * q.s11[phase];
        q.S22 += q.alpha[phase] * q.s22[phase];
        q.S12 += q.alpha[phase] * q.s12[phase];
    }

    //temperature from the total energy
    double coldEnergy = 0;
    for (int phase = 0; phase < 3; ++phase)
        if (q.alpha[phase] > 0)
            coldEnergy += U[MASS_RDX + phase] * multimaterialEOS::referenceEnergy(phase, q.density[phase], q.Y[0]);
    const double kinetic = 0.5 * q.rho * (q.u * q.u + q.v * q.v);
    q.T = (U[ENERGY] - kinetic - coldEnergy - elasticEnergy) / q.heatCapacity;
    if (!std::isfinite(q.T) || q.T <= 0) return fail("non-positive temperature");

    //volume-weighted pressure and frozen sound speed
    double dpdT = 0, dpdrho = 0;
    for (int phase = 0; phase < 3; ++phase)
    {
        if (q.alpha[phase] == 0) continue;
        const double rho = q.density[phase], a = q.alpha[phase];
        q.p += a * multimaterialEOS::pressureValue(phase, rho, q.T, q.Y[0]);
        dpdT += a * multimaterialEOS::pressureTemperatureDerivative(phase, rho, q.T, q.Y[0]);
        dpdrho += a * rho * multimaterialEOS::pressureDensityDerivative(phase, rho, q.T, q.Y[0]);
    }
    const double c2 = dpdrho / q.rho + q.T * dpdT * dpdT / (q.rho * q.heatCapacity);
    if (!std::isfinite(q.p) || !std::isfinite(c2) || c2 <= 0) return fail("invalid pressure or sound speed");

    //longitudinal wave speed: bulk + 4/3 shear stiffness
    q.c = std::sqrt(c2 + 4.0 / 3.0 * shearStiffness / q.rho);
    return true;
}

//primitive variables -> conserved state (uses alpha, density, T, u, v, Y, s, ep)
Vars computeConserved(const Primitive &q)
{
    Vars U = Vars::Zero();
    double rho = 0;
    for (int phase = 0; phase < 3; ++phase)
    {
        U[ALPHA_RDX + phase] = q.alpha[phase];
        if (q.alpha[phase] == 0) continue;
        const double mass = q.alpha[phase] * q.density[phase];
        U[MASS_RDX + phase] = mass;
        U[ENERGY] += mass * multimaterialEOS::internalEnergy(phase, q.density[phase], q.T, q.Y[0]);
        rho += mass;
    }
    for (int k = 0; k < 3; ++k)
        U[RHOY1 + k] = U[MASS_RDX] * q.Y[k];

    for (int phase = 0; phase < nSolids; ++phase)
    {
        const double mass = U[MASS_RDX + phase];
        U[S11_RDX + 3 * phase] = mass * q.s11[phase];
        U[S22_RDX + 3 * phase] = mass * q.s22[phase];
        U[S12_RDX + 3 * phase] = mass * q.s12[phase];
        U[EP_RDX + phase] = mass * q.ep[phase];
        const double G = shearModulus(phase);
        if (G > 0 && q.alpha[phase] > 0)
            U[ENERGY] += q.alpha[phase] * doubleContraction(q.s11[phase], q.s22[phase], q.s12[phase]) / (4.0 * G);
    }

    U[MOMX] = rho * q.u;
    U[MOMY] = rho * q.v;
    U[ENERGY] += 0.5 * rho * (q.u * q.u + q.v * q.v);
    return U;
}

//cell (i, j), including ghost cells outside the grid.
//open sides copy the nearest cell; walls mirror it (slip, adiabatic)
Vars sample(const State &state, int i, int j)
{
    const bool mirrorX = (i < 0 && wallLeft) || (i >= nx && wallRight);
    const bool mirrorY = (j < 0 && wallBottom) || (j >= ny && wallTop);

    int ii = std::clamp(i, 0, nx - 1), jj = std::clamp(j, 0, ny - 1);
    if (mirrorX) ii = i < 0 ? -i - 1 : 2 * nx - 1 - i;
    if (mirrorY) jj = j < 0 ? -j - 1 : 2 * ny - 1 - j;
    Vars U = getCell(state, index(ii, jj));

    //reflection flips the normal momentum and the sign of the shear stress
    if (mirrorX) U[MOMX] = -U[MOMX];
    if (mirrorY) U[MOMY] = -U[MOMY];
    if (mirrorX != mirrorY)
        for (int phase = 0; phase < nSolids; ++phase)
            U[S12_RDX + 3 * phase] = -U[S12_RDX + 3 * phase];
    return U;
}

//face state from the donor cell `center`, with THINC-sharpened volume fractions.
//The donor keeps its own phase densities, temperature, velocity, species and stresses.
bool thincFaceState(const std::array<Vars, 6> &cells, int center, bool rightEdge, Vars &face)
{
    Primitive q;
    if (!computePrimitive(cells[center], q)) return false;

    double sum = 0;
    for (int phase = 0; phase < 3; ++phase)
    {
        const int k = ALPHA_RDX + phase;
        const auto edges = THINC::reconstruct(cells[center - 1][k], cells[center][k], cells[center + 1][k], thincBeta);
        q.alpha[phase] = q.alpha[phase] == 0 ? 0 : (rightEdge ? edges.right : edges.left);
        sum += q.alpha[phase];
    }
    if (sum <= 0) return false;
    for (int phase = 0; phase < 3; ++phase) q.alpha[phase] /= sum;

    face = computeConserved(q);
    return true;
}

//high-order states on both sides of the face between cells[2] and cells[3].
//Away from interfaces (same volume fractions in the whole stencil) all fields use
//WENO5 with common weights; at interfaces the volume fractions use THINC.
//Returns false if a reconstructed state is not physical.
bool reconstructFace(const std::array<Vars, 6> &cells, Vars &left, Vars &right)
{
    bool sameFractions = true;
    for (int n = 0; n < 6; ++n)
        for (int phase = 0; phase < 3; ++phase)
            if (cells[n][ALPHA_RDX + phase] != cells[2][ALPHA_RDX + phase]) sameFractions = false;

    if (sameFractions)
    {
        std::array<std::array<double, 6>, nFields> stencil;
        for (int k = 0; k < nFields; ++k)
            for (int n = 0; n < 6; ++n) stencil[k][n] = cells[n][k];
        const auto traces = WENO::reconstructSystem<nFields>(stencil, true, MOMX);
        for (int k = 0; k < nFields; ++k)
        {
            left[k] = traces[k].left;
            right[k] = traces[k].right;
        }
        for (int phase = 0; phase < 3; ++phase)
            left[ALPHA_RDX + phase] = right[ALPHA_RDX + phase] = cells[2][ALPHA_RDX + phase];
    }
    else
    {
        if (!thincFaceState(cells, 2, true, left)) return false;
        if (!thincFaceState(cells, 3, false, right)) return false;
    }

    Primitive q;
    return computePrimitive(left, q) && computePrimitive(right, q);
}

struct Face
{
    Vars flux;
    double un = 0;      //normal velocity at the face
    double ut = 0;      //tangential velocity at the face
    bool fellBack = false;
};

//Rusanov flux through face (i, j).
//x-face (i, j) lies between cells (i-1, j) and (i, j); y-face between (i, j-1) and (i, j).
Face computeFace(const State &state, int i, int j, bool alongX, bool firstOrder)
{
    Face face;
    std::array<Vars, 6> cells;
    for (int n = 0; n < 6; ++n)
        cells[n] = alongX ? sample(state, i + n - 3, j) : sample(state, i, j + n - 3);

    //the full six-cell stencil has to be inside the grid for high order
    const bool interior = alongX ? (i >= 3 && i <= nx - 3) : (j >= 3 && j <= ny - 3);
    Vars left = cells[2], right = cells[3];
    if (!firstOrder && interior && !reconstructFace(cells, left, right))
    {
        left = cells[2];
        right = cells[3];
        face.fellBack = true;
    }

    Primitive qL, qR;
    computePrimitive(left, qL);
    computePrimitive(right, qR);

    //normal (n) and tangential (t) components
    const int momN = alongX ? MOMX : MOMY;
    const int momT = alongX ? MOMY : MOMX;
    const double unL = alongX ? qL.u : qL.v, unR = alongX ? qR.u : qR.v;
    const double utL = alongX ? qL.v : qL.u, utR = alongX ? qR.v : qR.u;
    const double SnnL = alongX ? qL.S11 : qL.S22, SnnR = alongX ? qR.S11 : qR.S22;

    //physical fluxes: advection, then traction -sigma.n on momentum and -sigma.n.v on energy
    Vars fluxL = unL * left, fluxR = unR * right;
    fluxL[momN] += qL.p - SnnL;
    fluxR[momN] += qR.p - SnnR;
    fluxL[momT] -= qL.S12;
    fluxR[momT] -= qR.S12;
    fluxL[ENERGY] += unL * (qL.p - SnnL) - utL * qL.S12;
    fluxR[ENERGY] += unR * (qR.p - SnnR) - utR * qR.S12;

    const double speed = std::max(std::abs(unL) + qL.c, std::abs(unR) + qR.c);
    face.flux = 0.5 * (fluxL + fluxR) - 0.5 * speed * (right - left);

    //heat conduction between the two cell centers
    if (conduction)
    {
        Primitive qWest, qEast;
        computePrimitive(cells[2], qWest);
        computePrimitive(cells[3], qEast);
        const double spacing = alongX ? dx : dy;
        face.flux[ENERGY] -= params::k * (qEast.T - qWest.T) / spacing;
    }

    face.un = 0.5 * (unL + unR);
    face.ut = 0.5 * (utL + utR);
    return face;
}

//1/dt allowed by one cell: waves, conduction and chemistry
double inverseTimeStep(const Primitive &q)
{
    double rate = ((std::abs(q.u) + q.c) / dx + (std::abs(q.v) + q.c) / dy) / cfl;
    if (conduction)
        rate += 2.0 * params::k / q.heatCapacity * (1.0 / (dx * dx) + 1.0 / (dy * dy)) / cfl;

    if (reactions && q.alpha[RDX] > 0)
    {
        const double rdxMass = q.alpha[RDX] * q.density[RDX];
        const double r1 = reactionRates::computeArrhenius(q.T, params::R, params::E1, params::Z1);
        const double r2 = reactionRates::computeArrhenius(q.T, params::R, params::E2, params::Z2);
        const double rate1 = r1 * q.Y[0] * rdxMass, rate2 = r2 * q.Y[1] * rdxMass;
        const double heat = (params::rho_init / q.density[RDX]) *
                            (reactionRates::computeReactionHeat(params::a1, params::b1, q.T) * rate1 +
                             reactionRates::computeReactionHeat(params::a2, params::b2, q.T) * rate2);
        const double referenceChange = rate1 * std::abs(multimaterialEOS::rdxReferenceEnergy(q.density[RDX], false) -
                                                        multimaterialEOS::rdxReferenceEnergy(q.density[RDX], true));
        rate = std::max(rate, std::max(r1, r2) / 0.08);
        rate = std::max(rate, (std::abs(heat) + referenceChange) / (0.05 * q.heatCapacity * q.T));
    }
    return rate;
}

double stableTimeStep(const State &state)
{
    double maxRate = 0;
#pragma omp parallel for reduction(max : maxRate)
    for (int cell = 0; cell < nx * ny; ++cell)
    {
        Primitive q;
        computePrimitive(getCell(state, cell), q);
        maxRate = std::max(maxRate, inverseTimeStep(q));
    }
    return 1.0 / maxRate;
}

struct Residual
{
    State rate;
    double maxDt = 0;       //largest stable dt for this state
    long fallbackFaces = 0;
};

//right-hand side dU/dt of every field.
//firstOrder[cell] = 1 forces first-order fluxes on all faces of that cell.
Residual assembleResidual(const State &state, const std::vector<char> &firstOrder)
{
    const int nCells = nx * ny;
    Residual R;
    R.rate.assign(nFields, Eigen::VectorXd::Zero(nCells));

    //primitive variables of every cell
    std::vector<Primitive> prim(nCells);
#pragma omp parallel for
    for (int cell = 0; cell < nCells; ++cell)
        computePrimitive(getCell(state, cell), prim[cell]);

    //fluxes through every face
    std::vector<Face> xFaces((nx + 1) * ny), yFaces(nx * (ny + 1));
#pragma omp parallel for
    for (int f = 0; f < (nx + 1) * ny; ++f)
    {
        const int i = f % (nx + 1), j = f / (nx + 1);
        const bool low = (i > 0 && firstOrder[index(i - 1, j)]) || (i < nx && firstOrder[index(i, j)]);
        xFaces[f] = computeFace(state, i, j, true, low);
    }
#pragma omp parallel for
    for (int f = 0; f < nx * (ny + 1); ++f)
    {
        const int i = f % nx, j = f / nx;
        const bool low = (j > 0 && firstOrder[index(i, j - 1)]) || (j < ny && firstOrder[index(i, j)]);
        yFaces[f] = computeFace(state, i, j, false, low);
    }

    //update of every cell
#pragma omp parallel for
    for (int cell = 0; cell < nCells; ++cell)
    {
        const int i = cell % nx, j = cell / nx;
        const Face &west = xFaces[j * (nx + 1) + i];
        const Face &east = xFaces[j * (nx + 1) + i + 1];
        const Face &south = yFaces[j * nx + i];
        const Face &north = yFaces[(j + 1) * nx + i];
        const Primitive &q = prim[cell];

        //transport
        for (int k = 0; k < nFields; ++k)
            R.rate[k][cell] = (west.flux[k] - east.flux[k]) / dx + (south.flux[k] - north.flux[k]) / dy;

        //velocity gradient from the face velocities
        const double dudx = (east.un - west.un) / dx;
        const double dvdx = (east.ut - west.ut) / dx;
        const double dvdy = (north.un - south.un) / dy;
        const double dudy = (north.ut - south.ut) / dy;
        const double divergence = dudx + dvdy;

        //volume fractions: d(alpha)/dt + div(alpha v) = alpha div(v)
        for (int phase = 0; phase < 3; ++phase)
            R.rate[ALPHA_RDX + phase][cell] += state[ALPHA_RDX + phase][cell] * divergence;

        //deviatoric stress: 2 G dev(D) + W s - s W
        const double D11 = dudx - divergence / 3.0;
        const double D22 = dvdy - divergence / 3.0;
        const double D12 = 0.5 * (dudy + dvdx);
        const double W12 = 0.5 * (dudy - dvdx);
        for (int phase = 0; phase < nSolids; ++phase)
        {
            const double mass = state[MASS_RDX + phase][cell];
            if (mass <= 0) continue;
            const double G = shearModulus(phase);
            const double s11 = q.s11[phase], s22 = q.s22[phase], s12 = q.s12[phase];
            R.rate[S11_RDX + 3 * phase][cell] += mass * (2.0 * G * D11 + 2.0 * W12 * s12);
            R.rate[S22_RDX + 3 * phase][cell] += mass * (2.0 * G * D22 - 2.0 * W12 * s12);
            R.rate[S12_RDX + 3 * phase][cell] += mass * (2.0 * G * D12 + W12 * (s22 - s11));
        }

        //chemistry (RDX only)
        if (reactions && state[MASS_RDX][cell] > 0)
        {
            const double r1 = reactionRates::computeArrhenius(q.T, params::R, params::E1, params::Z1);
            const double r2 = reactionRates::computeArrhenius(q.T, params::R, params::E2, params::Z2);
            const double rate1 = r1 * state[RHOY1][cell];
            const double rate2 = r2 * state[RHOY2][cell];
            const double Q1 = reactionRates::computeReactionHeat(params::a1, params::b1, q.T);
            const double Q2 = reactionRates::computeReactionHeat(params::a2, params::b2, q.T);
            R.rate[RHOY1][cell] -= rate1;
            R.rate[RHOY2][cell] += rate1 - rate2;
            R.rate[RHOY3][cell] += rate2;
            R.rate[ENERGY][cell] += (params::rho_init / q.density[RDX]) * (Q1 * rate1 + Q2 * rate2);
        }
    }

    //stable time step and fallback count
    double maxRate = 0;
    for (int cell = 0; cell < nCells; ++cell) maxRate = std::max(maxRate, inverseTimeStep(prim[cell]));
    R.maxDt = 1.0 / maxRate;
    for (const Face &f : xFaces) R.fallbackFaces += f.fellBack;
    for (const Face &f : yFaces) R.fallbackFaces += f.fellBack;
    return R;
}

//the end supports do not move; their kinetic energy is removed
void applySupports(State &state)
{
    for (int cell : supportCells)
    {
        const double rho = state[MASS_RDX][cell] + state[MASS_PS][cell] + state[MASS_AIR][cell];
        const double mx = state[MOMX][cell], my = state[MOMY][cell];
        state[ENERGY][cell] -= 0.5 * (mx * mx + my * my) / rho;
        state[MOMX][cell] = 0;
        state[MOMY][cell] = 0;
    }
}

std::vector<int> findInvalidCells(const State &state)
{
    std::vector<char> bad(nx * ny, 0);
#pragma omp parallel for
    for (int cell = 0; cell < nx * ny; ++cell)
    {
        Primitive q;
        bad[cell] = !computePrimitive(getCell(state, cell), q);
    }
    std::vector<int> cells;
    for (int cell = 0; cell < nx * ny; ++cell)
        if (bad[cell]) cells.push_back(cell);
    return cells;
}

//one line describing why a cell is not physical, for error messages
std::string describeCell(const State &state, int cell)
{
    Primitive q;
    std::string reason = "valid";
    computePrimitive(getCell(state, cell), q, &reason);
    std::ostringstream text;
    text << std::setprecision(6) << "cell (" << cell % nx << ", " << cell / nx << ") at x = " << (cell % nx + 0.5) * dx
         << " nm, y = " << (cell / nx + 0.5) * dy << " nm: " << reason << "; alpha (rdx ps air) = "
         << state[ALPHA_RDX][cell] << ' ' << state[ALPHA_PS][cell] << ' ' << state[ALPHA_AIR][cell]
         << ", alpha*rho = " << state[MASS_RDX][cell] << ' ' << state[MASS_PS][cell] << ' ' << state[MASS_AIR][cell];
    return text.str();
}

//SSP-RK3 step. On failure the state is left unchanged and the function returns false;
//badCells then lists the cells that became unphysical (empty if dt was too large).
bool RK3Step(State &state, double dt, const std::vector<char> &firstOrder, std::vector<int> &badCells,
             long &fallbackFaces, std::string &failure)
{
    const State state_n = state;
    State state1 = state_n, state2 = state_n;

    //first stage
    Residual R = assembleResidual(state_n, firstOrder);
    if (dt > R.maxDt * (1 + 1e-12))
    {
        failure = "time step above the stable limit of an RK stage";
        return false;
    }
    fallbackFaces += R.fallbackFaces;
    for (int k = 0; k < nFields; ++k)
        state1[k] = state_n[k] + dt * R.rate[k];
    applySupports(state1);
    badCells = findInvalidCells(state1);
    if (!badCells.empty())
    {
        failure = std::to_string(badCells.size()) + " invalid cell(s), first " + describeCell(state1, badCells[0]);
        return false;
    }

    //second stage
    R = assembleResidual(state1, firstOrder);
    if (dt > R.maxDt * (1 + 1e-12))
    {
        failure = "time step above the stable limit of an RK stage";
        return false;
    }
    fallbackFaces += R.fallbackFaces;
    for (int k = 0; k < nFields; ++k)
        state2[k] = 0.75 * state_n[k] + 0.25 * state1[k] + 0.25 * dt * R.rate[k];
    applySupports(state2);
    badCells = findInvalidCells(state2);
    if (!badCells.empty())
    {
        failure = std::to_string(badCells.size()) + " invalid cell(s), first " + describeCell(state2, badCells[0]);
        return false;
    }

    //final stage
    R = assembleResidual(state2, firstOrder);
    if (dt > R.maxDt * (1 + 1e-12))
    {
        failure = "time step above the stable limit of an RK stage";
        return false;
    }
    fallbackFaces += R.fallbackFaces;
    State next = state_n;
    for (int k = 0; k < nFields; ++k)
        next[k] = (1.0 / 3.0) * state_n[k] + (2.0 / 3.0) * state2[k] + (2.0 / 3.0) * dt * R.rate[k];
    applySupports(next);
    badCells = findInvalidCells(next);
    if (!badCells.empty())
    {
        failure = std::to_string(badCells.size()) + " invalid cell(s), first " + describeCell(next, badCells[0]);
        return false;
    }

    state = next;
    return true;
}

//radial return of every solid phase that is above its yield stress.
//Solves  trialStress - 3 G dLambda = yield(ep + dLambda, dLambda / dt, T)  by bisection,
//then scales the deviator back to the yield surface and adds dLambda to ep.
//Returns the number of (cell, phase) pairs that yielded.
long applyPlasticity(State &state, double dt)
{
    long yielded = 0;
#pragma omp parallel for reduction(+ : yielded)
    for (int cell = 0; cell < nx * ny; ++cell)
    {
        Primitive q;
        if (!computePrimitive(getCell(state, cell), q)) continue;

        for (int phase = 0; phase < nSolids; ++phase)
        {
            const double mass = state[MASS_RDX + phase][cell];
            const double G = shearModulus(phase);
            if (mass <= 0 || G <= 0) continue;

            const double ep = std::max(q.ep[phase], 0.0);
            const double trialStress = vonMises(q.s11[phase], q.s22[phase], q.s12[phase]);
            if (trialStress <= yieldStress(phase, ep, 0.0, q.T)) continue;

            //plastic multiplier: the yield function decreases with dLambda
            double low = 0.0, high = trialStress / (3.0 * G);
            for (int iter = 0; iter < 60; ++iter)
            {
                const double dLambda = 0.5 * (low + high);
                const double f = trialStress - 3.0 * G * dLambda - yieldStress(phase, ep + dLambda, dLambda / dt, q.T);
                if (f > 0) low = dLambda;
                else high = dLambda;
            }
            const double dLambda = high;
            const double scale = (trialStress - 3.0 * G * dLambda) / trialStress;

            state[S11_RDX + 3 * phase][cell] *= scale;
            state[S22_RDX + 3 * phase][cell] *= scale;
            state[S12_RDX + 3 * phase][cell] *= scale;
            state[EP_RDX + phase][cell] += mass * dLambda;
            ++yielded;
        }
    }
    return yielded;
}

//flag the bad cells for first order; a bad cell that is already flagged also flags
//its 3x3 neighbourhood. Returns false if nothing new was flagged.
bool flagFirstOrder(const std::vector<int> &badCells, std::vector<char> &firstOrder)
{
    bool added = false;
    for (int cell : badCells)
    {
        if (!firstOrder[cell])
        {
            firstOrder[cell] = 1;
            added = true;
            continue;
        }
        const int i = cell % nx, j = cell / nx;
        for (int dj = -1; dj <= 1; ++dj)
            for (int di = -1; di <= 1; ++di)
            {
                const int ni = i + di, nj = j + dj;
                if (ni < 0 || nj < 0 || ni >= nx || nj >= ny) continue;
                if (!firstOrder[index(ni, nj)]) added = true;
                firstOrder[index(ni, nj)] = 1;
            }
    }
    return added;
}

const std::vector<std::string> outputNames = {
    "Y1", "Y2", "Y3", "T", "rho", "vx", "vy", "rhoE", "p",
    "alpha_rdx", "alpha_ps", "alpha_pore",
    "alpha_rho_rdx", "alpha_rho_ps", "alpha_rho_pore",
    "rho_rdx", "rho_ps", "rho_pore",
    "s11", "s22", "s12", "sigma_eq", "ep_rdx", "ep_ps",
    "disp_x", "disp_y"};

std::vector<Eigen::VectorXd> outputFields(const State &state, double time)
{
    std::vector<Eigen::VectorXd> fields(outputNames.size(), Eigen::VectorXd::Zero(nx * ny));
#pragma omp parallel for
    for (int cell = 0; cell < nx * ny; ++cell)
    {
        Primitive q;
        computePrimitive(getCell(state, cell), q);
        for (int k = 0; k < 3; ++k) fields[k][cell] = q.Y[k];
        fields[3][cell] = q.T;
        fields[4][cell] = q.rho;
        fields[5][cell] = q.u + pistonVelocity; //laboratory frame
        fields[6][cell] = q.v;
        fields[7][cell] = state[ENERGY][cell];
        fields[8][cell] = q.p;
        for (int phase = 0; phase < 3; ++phase)
        {
            fields[9 + phase][cell] = q.alpha[phase];
            fields[12 + phase][cell] = state[MASS_RDX + phase][cell];
            fields[15 + phase][cell] = q.density[phase];
        }
        fields[18][cell] = q.S11;
        fields[19][cell] = q.S22;
        fields[20][cell] = q.S12;
        fields[21][cell] = vonMises(q.S11, q.S22, q.S12);
        fields[22][cell] = q.ep[RDX];
        fields[23][cell] = q.ep[PS];
        //the grid moves with the piston in the laboratory frame
        fields[24][cell] = pistonVelocity * time;
        fields[25][cell] = 0;
    }
    return fields;
}

//The Rusanov dissipation leaks a little of every phase into each neighbour at every
//stage, so traces (down to subnormal values) spread over the whole domain. The WENO5
//switch in reconstruct() needs identical fractions across the stencil, so without this
//cleanup nearly every face falls back to first order within a few steps.
//In each cell, a phase with 0 < alpha < traceFraction is merged into the dominant phase:
//its volume and partial mass move to that phase, whose mass-weighted fields (RDX
//species, solid stress and plastic strain) are scaled so their specific values stay
//the same; the trace phase's own stress and plastic strain are dropped. Momentum and
//total energy are untouched and the dominant fraction is set to exactly
//1 - (other fractions). Total mass, momentum and energy are conserved exactly; the
//per-phase mass moved is returned (per volume). It also clips round-off negatives in
//the RDX species (the difference goes to the largest species, so they still sum to
//the RDX mass exactly); that part runs even with traceFraction = 0.
std::array<double, 3> removeTracePhases(State &state)
{
    std::array<double, 3> moved{};
    for (int cell = 0; cell < nx * ny; ++cell)
    {
        int dominant = 0;
        for (int phase = 1; phase < 3; ++phase)
            if (state[ALPHA_RDX + phase][cell] > state[ALPHA_RDX + dominant][cell]) dominant = phase;
        bool changed = false;
        for (int phase = 0; phase < 3; ++phase)
        {
            const double alpha = state[ALPHA_RDX + phase][cell];
            if (phase == dominant || !(alpha > 0) || alpha >= traceFraction) continue;  //none if traceFraction = 0
            const double mass = state[MASS_RDX + phase][cell];
            const double dominantMass = state[MASS_RDX + dominant][cell];
            if (dominantMass > 0)
            {
                const double scale = (dominantMass + mass) / dominantMass;
                if (dominant == RDX)
                    for (int k = 0; k < 3; ++k) state[RHOY1 + k][cell] *= scale;
                if (dominant < nSolids)
                {
                    for (int c = 0; c < 3; ++c) state[S11_RDX + 3 * dominant + c][cell] *= scale;
                    state[EP_RDX + dominant][cell] *= scale;
                }
            }
            if (phase == RDX)
                for (int k = 0; k < 3; ++k) state[RHOY1 + k][cell] = 0;
            if (phase < nSolids)
            {
                for (int c = 0; c < 3; ++c) state[S11_RDX + 3 * phase + c][cell] = 0;
                state[EP_RDX + phase][cell] = 0;
            }
            state[MASS_RDX + dominant][cell] += mass;
            state[MASS_RDX + phase][cell] = 0;
            state[ALPHA_RDX + phase][cell] = 0;
            moved[phase] -= mass;
            moved[dominant] += mass;
            changed = true;
        }
        //exact closure of the fractions (also removes round-off drift in pure cells)
        double others = 0;
        for (int phase = 0; phase < 3; ++phase)
            if (phase != dominant) others += state[ALPHA_RDX + phase][cell];
        if (changed || std::abs(state[ALPHA_RDX + dominant][cell] + others - 1) <= 1e-12)
            state[ALPHA_RDX + dominant][cell] = 1 - others;

        //species: clip round-off negatives, close the sum on the largest species
        const double rdxMass = state[MASS_RDX][cell];
        if (rdxMass > 0)
        {
            int largest = 0;
            double sum = 0;
            for (int k = 0; k < 3; ++k)
            {
                double &Yk = state[RHOY1 + k][cell];
                if (Yk < 0) Yk = 0;
                if (Yk > state[RHOY1 + largest][cell]) largest = k;
                sum += Yk;
            }
            state[RHOY1 + largest][cell] += rdxMass - sum;
        }
    }
    return moved;
}

int main(int argc, char *argv[])
{
    if (argc < 9)
    {
        std::cerr << "Usage: " << argv[0]
                  << " Nx Ny Lx Ly interval output.e steps stride --density-map x_y_rho_phase.csv\n"
                     "  [--piston velocity]        left wall pushes into the material (piston frame)\n"
                     "  [--velocity vx vy]         initial velocity of all material\n"
                     "  [--rdx-velocity vx vy]     initial velocity of the RDX only (applied last)\n"
                     "  [--boundaries LRBT]        w = slip wall, o = open, for left right bottom top (default wwww)\n"
                     "  [--open-boundaries]        same as --boundaries oooo\n"
                     "  [--fixed-ps-ends width] [--thinc-beta value] [--cfl value]\n"
                     "  [--high-order-halvings n] [--trace-fraction value] [--no-chemistry] [--no-conduction]\n";
        return 1;
    }

    try
    {
        nx = std::stoi(argv[1]);
        ny = std::stoi(argv[2]);
        Lx = std::stod(argv[3]);
        Ly = std::stod(argv[4]);
        const double interval = std::stod(argv[5]);
        const std::string outputPath = argv[6];
        const int steps = std::stoi(argv[7]);
        const int stride = std::stoi(argv[8]);
        dx = Lx / nx;
        dy = Ly / ny;

        std::string mapPath;
        double vx = 0, vy = 0, supportWidth = 0;
        double rdxVx = 0, rdxVy = 0;
        bool rdxVelocity = false;
        std::string boundaries = "wwww";
        for (int a = 9; a < argc; ++a)
        {
            const std::string option = argv[a];
            if (option == "--density-map" && a + 1 < argc) mapPath = argv[++a];
            else if (option == "--rdx-velocity" && a + 2 < argc)
            {
                rdxVx = std::stod(argv[++a]);
                rdxVy = std::stod(argv[++a]);
                rdxVelocity = true;
            }
            else if (option == "--velocity" && a + 2 < argc)
            {
                vx = std::stod(argv[++a]);
                vy = std::stod(argv[++a]);
            }
            else if (option == "--piston" && a + 1 < argc) pistonVelocity = std::stod(argv[++a]);
            else if (option == "--boundaries" && a + 1 < argc) boundaries = argv[++a];
            else if (option == "--open-boundaries") boundaries = "oooo";
            else if (option == "--fixed-ps-ends" && a + 1 < argc) supportWidth = std::stod(argv[++a]);
            else if (option == "--thinc-beta" && a + 1 < argc) thincBeta = std::stod(argv[++a]);
            else if (option == "--cfl" && a + 1 < argc) cfl = std::stod(argv[++a]);
            else if (option == "--high-order-halvings" && a + 1 < argc) highOrderHalvings = std::stoi(argv[++a]);
            else if (option == "--trace-fraction" && a + 1 < argc) traceFraction = std::stod(argv[++a]);
            else if (option == "--no-chemistry") reactions = false;
            else if (option == "--no-conduction") conduction = false;
            else throw std::invalid_argument("Unknown or incomplete option: " + option);
        }

        if (nx < 6 || ny < 6 || !(Lx > 0) || !(Ly > 0) || !(interval > 0) || steps < 0 || stride < 1)
            throw std::invalid_argument("Invalid grid or time parameters");
        if (!(thincBeta > 0) || thincBeta > 10 || !(cfl > 0) || cfl > 1 || highOrderHalvings < 0)
            throw std::invalid_argument("Invalid --thinc-beta, --cfl or --high-order-halvings");
        if (!(traceFraction >= 0) || traceFraction > 1e-3)
            throw std::invalid_argument("--trace-fraction must be between 0 and 1e-3");
        if (mapPath.empty())
            throw std::invalid_argument("--density-map is required");
        if (supportWidth < 0 || supportWidth >= Lx / 2)
            throw std::invalid_argument("Invalid support width");
        if (boundaries.size() != 4 || boundaries.find_first_not_of("wo") != std::string::npos)
            throw std::invalid_argument("--boundaries needs four letters, w or o, for left right bottom top");
        wallLeft = boundaries[0] == 'w';
        wallRight = boundaries[1] == 'w';
        wallBottom = boundaries[2] == 'w';
        wallTop = boundaries[3] == 'w';
        if (pistonVelocity != 0)
        {
            //piston frame: wall on the left, material flows in from the open right side
            if (!wallLeft || wallRight)
                throw std::invalid_argument("--piston needs a wall on the left and an open right side (e.g. --boundaries wooo)");
            vx -= pistonVelocity;
            if (rdxVelocity) rdxVx -= pistonVelocity;
        }

        //initial state from the phase map: pure phases at 300 K, stress free
        State state(nFields, Eigen::VectorXd::Zero(nx * ny));
        const auto map = densityMap2D::read(mapPath, true);
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
            {
                const int cell = index(i, j);
                const double x = (i + 0.5) * dx, y = (j + 0.5) * dy;
                const int phase = static_cast<int>(map.phaseAt(x, y));
                const double density = map.at(x, y);
                if (!(density > 0)) throw std::invalid_argument("Every phase needs a positive map density");

                Primitive q;
                q.alpha[phase] = 1;
                q.density[phase] = density;
                q.T = 300;
                q.Y[0] = phase == RDX ? 1 : 0;
                q.u = (phase == RDX && rdxVelocity) ? rdxVx : vx;
                q.v = (phase == RDX && rdxVelocity) ? rdxVy : vy;
                const Vars U = computeConserved(q);
                for (int k = 0; k < nFields; ++k) state[k][cell] = U[k];

                if (phase == PS && (x < supportWidth || x > Lx - supportWidth))
                    supportCells.push_back(cell);
            }
        if (supportWidth > 0 && supportCells.empty())
            throw std::invalid_argument("No PS cells found in the end supports");
        if (!findInvalidCells(state).empty())
            throw std::invalid_argument("Invalid initial state");

#ifdef _OPENMP
        std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
#endif
        std::cout << "Boundaries (left right bottom top) " << boundaries << ", piston " << pistonVelocity
                  << " nm/ns, support cells " << supportCells.size() << ", G_rdx " << params::G_rdx << " GPa, G_ps " << params::G_ps << " GPa\n";

        ExodusOutput exodus(outputPath, nx, ny, dx, dy, outputNames, 0.5 * dx, 0.5 * dy);
        std::ofstream diagnostics(outputPath + ".diagnostics.csv");
        diagnostics << std::setprecision(17)
                    << "time_ns,rdx_mass,ps_mass,air_mass,total_energy,Tmin,Tmax,max_sigma_eq,"
                       "internal_steps,retries,first_order_steps,fallback_faces,yielded,trace_mass_moved\n";

        long internalSteps = 0, retries = 0, firstOrderSteps = 0, fallbackFaces = 0, yielded = 0;
        double traceMoved = 0; //total phase mass moved by removeTracePhases
        auto save = [&](double time)
        {
            const auto fields = outputFields(state, time);
            exodus.write(time, fields);

            const double volume = dx * dy;
            diagnostics << time << ',' << state[MASS_RDX].sum() * volume << ',' << state[MASS_PS].sum() * volume
                        << ',' << state[MASS_AIR].sum() * volume << ',' << state[ENERGY].sum() * volume << ','
                        << fields[3].minCoeff() << ',' << fields[3].maxCoeff() << ',' << fields[21].maxCoeff()
                        << ',' << internalSteps << ',' << retries << ',' << firstOrderSteps << ',' << fallbackFaces
                        << ',' << yielded << ',' << traceMoved << std::endl;
            std::cout << "time " << time << " ns, T=[" << fields[3].minCoeff() << ',' << fields[3].maxCoeff()
                      << "], max sigma_eq " << fields[21].maxCoeff() << " GPa\n";
        };

        double time = 0;
        save(time);
        std::vector<char> firstOrder(nx * ny, 0);
        for (int step = 1; step <= steps; ++step)
        {
            const double target = step * interval;
            while (time < target)
            {
                double dt = std::min(target - time, 0.9 * stableTimeStep(state));
                std::fill(firstOrder.begin(), firstOrder.end(), 0);
                bool accepted = false;
                std::string failure;

                //1) retry the high-order step with dt/2, dt/4, ...
                //2) then use first-order fluxes around the cells that failed (widening if needed)
                for (int attempt = 0; attempt < 40 && time + dt > time; ++attempt)
                {
                    std::vector<int> badCells;
                    if (RK3Step(state, dt, firstOrder, badCells, fallbackFaces, failure))
                    {
                        accepted = true;
                        break;
                    }
                    ++retries;
                    const bool halve = attempt < highOrderHalvings || badCells.empty() ||
                                       !flagFirstOrder(badCells, firstOrder);
                    if (halve) dt *= 0.5;
                }
                if (!accepted)
                {
                    std::ostringstream message;
                    message << std::setprecision(6) << "Cannot advance at time " << time << " ns (last dt " << dt
                            << " ns, " << std::count(firstOrder.begin(), firstOrder.end(), 1)
                            << " cells first order): " << failure;
                    throw std::runtime_error(message.str());
                }

                yielded += applyPlasticity(state, dt);
                for (double m : removeTracePhases(state)) traceMoved += std::max(m, 0.0) * dx * dy;
                time += dt;
                ++internalSteps;
                if (std::find(firstOrder.begin(), firstOrder.end(), 1) != firstOrder.end()) ++firstOrderSteps;
            }
            if (step % stride == 0 || step == steps) save(time);
        }
        exodus.close();
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
