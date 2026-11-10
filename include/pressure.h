#pragma once
#include <cmath>
#include "modelParams.h"

namespace pressure{

    inline double value(double rho, double T, bool reacted){
        //computes temperature dependent JWL pressure
        const double A = reacted ? params::A_reacted : params::A_unreacted;
        const double B = reacted ? params::B_reacted : params::B_unreacted;
        const double R1 = reacted ? params::R1_reacted : params::R1_unreacted;
        const double R2 = reacted ? params::R2_reacted : params::R2_unreacted;
        const double omega = reacted ? params::omega_reacted : params::omega_unreacted;
        const double cv = params::cv;
        const double J = params::rho_init / rho;
        const double p = A * std::exp(-R1 * J) + B * std::exp(-R2 * J) + omega * rho * cv * (T - 300);
        const double p0 = A * std::exp(-R1 * 1.0) + B * std::exp(-R2 * 1.0);
        return -(p - p0);
    }

    inline double dT(double rho, double T, bool reacted){
        const double omega = reacted ? params::omega_reacted : params::omega_unreacted;
        const double cv = params::cv;
        return - omega * rho * cv;
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
        const double &dpdT = dT(rho, T, reacted);

        //construct squared sound speed
        const double c2 = dpdrho + (T / std::pow(rho, 2) / params::cv) * std::pow(dpdT, 2);
        return c2;
    }
}