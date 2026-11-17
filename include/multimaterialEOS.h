#pragma once
#include <array>
#include <cmath>
#include <stdexcept>
#include "modelParams.h"
#include "pressureRDX.h"
#include "pressurePS.h"
#include "pressureAir.h"

// Solver units: nm, ns, GPa; density is kg/m^3 * 1e-9.
// Specific energy and heat capacity retain J/kg and J/(kg K) numerical values.
namespace multimaterialEOS
{
enum Phase
{
    RDX = 0,
    PS = 1,
    Air = 2
};
const double rhoPS = params::rho_ps;
const double cvPS = params::cv_ps;
const double cvAir = params::cv_air;
const double gasConstantAir = params::R_air;
constexpr double defaultAirDensity = 1.2e-9;
inline double heatCapacity(int phase)
{
    return phase == RDX ? params::cv : phase == PS ? cvPS : cvAir;
}
// All pressures here are compression-positive. The legacy RDX value() is stress.
inline double pressureValue(int phase, double rho, double T, double Y1 = 1)
{
    if (phase == RDX)
        return -Y1 * pressureRDX::value(rho, T, false) -
               (1 - Y1) * pressureRDX::value(rho, T, true);
    if (phase == PS)
        return -pressurePS::value(rho, T);
    return pressureAir::value(rho, T);
}
inline double pressureDensityDerivative(int phase, double rho, double T, double Y1 = 1)
{
    if (phase == RDX)
        return Y1 * pressureRDX::drho(rho, T, false) + (1 - Y1) * pressureRDX::drho(rho, T, true);
    if (phase == PS)
        return -pressurePS::drho(rho, T);
    return pressureAir::drho(rho, T);
}
inline double pressureTemperatureDerivative(int phase, double rho, double T, double Y1 = 1)
{
    if (phase == RDX)
        return -Y1 * pressureRDX::dT(rho, T, false) - (1 - Y1) * pressureRDX::dT(rho, T, true);
    return phase == PS ? -pressurePS::dT(rho, T) : pressureAir::dT(rho, T);
}
inline double rdxReferenceEnergy(double rho, bool reacted)
{
    const double compression = 1 - params::rho_init / rho;
    const double omega = reacted ? params::omega_reacted : params::omega_unreacted;
    const double A = reacted ? params::A_reacted : params::A_unreacted;
    const double B = reacted ? params::B_reacted : params::B_unreacted;
    const double R1 = reacted ? params::R1_reacted : params::R1_unreacted;
    const double R2 = reacted ? params::R2_reacted : params::R2_unreacted;
    auto work = [&](double a, double r)
    { return a * std::exp(-r) * (std::expm1(r * compression) / r - compression); };
    return (work(A, R1) + work(B, R2)) / params::rho_init +
           omega * params::cv * 300 * std::log(params::rho_init / rho);
}
inline double referenceEnergy(int phase, double rho, double Y1 = 1)
{
    if (phase == RDX)
        return Y1 * rdxReferenceEnergy(rho, false) + (1 - Y1) * rdxReferenceEnergy(rho, true);
    // de_ref/drho = (p - T*dp/dT)/rho^2; zero at the PS reference density.
    if (phase == PS)
        return .5 * params::K / rhoPS * std::pow(1 - rhoPS / rho, 2);
    return 0;
}
inline double internalEnergy(int phase, double rho, double T, double Y1 = 1)
{
    return heatCapacity(phase) * T + referenceEnergy(phase, rho, Y1);
}
} // namespace multimaterialEOS
