#pragma once
#include <algorithm>
#include <cmath>
#include "modelParams.h"
#include "pressureRDX.h"

namespace plasticityRDX{

    inline double yield(double ep, double increment, double T){
        //define a johnson-cook yield
        const double A = params::A_rdx;
        const double B = params::B_rdx;
        const double n = params::n_rdx;
        const double C = params::C_rdx;
        const double Tm = params::Tm_rdx;
        const double m_rdx = params::m_rdx;
        const double T0 = 300.0;

        //strain hardening
        const double strainHardening = A + B * std::pow(ep, n);

        //strain rate hardening, relative to the reference rate ref_ep (1/ns);
        //rates below the reference do not soften the material
        const double strainRate = std::max(increment, params::ref_ep);
        const double strainRateHardening = 1.0 + C * std::log(strainRate / params::ref_ep);

        //thermal softening; theta is clamped to [0, 1] (no yield above the melt temperature)
        const double theta = std::min(std::max((T - T0) / (Tm - T0), 0.0), 1.0);
        const double thermalSoftening = 1.0 - std::pow(theta, m_rdx);

        return strainHardening * strainRateHardening * thermalSoftening;
    }

    inline double yieldFunction(double eff_sigma, double ep, double increment, double T){
        //compute the J2 yield function
        return eff_sigma - yield(ep, increment, T);
    }

    inline double drho(double rho, double T, bool reacted){
        const double A = reacted ? params::A_reacted : params::A_unreacted;
        const double B = reacted ? params::B_reacted : params::B_unreacted;
        const double R1 = reacted ? params::R1_reacted : params::R1_unreacted;
        const double R2 = reacted ? params::R2_reacted : params::R2_unreacted;
        const double omega = reacted ? params::omega_reacted : params::omega_unreacted;
        const double cv = params::cv;
        const double J = params::rho_init / rho;
        const double dpdrho = (J / rho) * (A * R1 * std::exp(-R1 * J) + B * R2 * std::exp(-R2 * J)) + omega * cv * (T - 300);
        return dpdrho;
    }

    inline double c2(double rho, double T, bool reacted){
        const double A = reacted ? params::A_reacted : params::A_unreacted;
        const double B = reacted ? params::B_reacted : params::B_unreacted;
        const double R1 = reacted ? params::R1_reacted : params::R1_unreacted;
        const double R2 = reacted ? params::R2_reacted : params::R2_unreacted;
        const double omega = reacted ? params::omega_reacted : params::omega_unreacted;
        const double cv = params::cv;
        const double J = params::rho_init / rho;
        
        const double &dpdrho = drho(rho, T, reacted);
        const double &dpdT = pressureRDX::dT(rho, T, reacted);

        //construct squared sound speed
        const double c2 = dpdrho + (T / std::pow(rho, 2) / params::cv) * std::pow(dpdT, 2);
        return c2;
    }
}