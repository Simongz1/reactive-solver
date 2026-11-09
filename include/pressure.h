#pragma once
#include <cmath>
#include "modelParams.h"

namespace pressure{
    inline double value(double rho, double T){
        //computes temperature dependent JWL pressure
        const double A = params::A;
        const double B = params::B;
        const double R1 = params::R1;
        const double R2 = params::R2;
        const double omega = params::omega;
        const double cv = params::cv;
        const double ratio = params::rho_init / rho;
        const double p = A * std::exp(-R1 * ratio) + B * std::exp(-R2 * ratio) + omega * rho * cv * (T - 300);
        const double p0 = A * std::exp(-R1 * 1.0) + B * std::exp(-R2 * 1.0);

        return -(p - p0);
    }
    inline double dT(double rho, double T){
        const double omega = params::omega;
        const double cv = params::cv;
        return - omega * rho * cv;
    }
}