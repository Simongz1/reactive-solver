#pragma once
#include <cmath>
#include "modelParams.h"

namespace pressurePS{

    inline double value(double rho, double T){
        //computes the pressure for polystyrene
        const double K = params::K;
        const double J = params::rho_ps / rho;

        //compute pressure
        return K * (J - 1.0);
    }

    inline double dT(double rho, double T){
        return 0.0;
    }

    inline double drho(double rho, double T){
        const double K = params::K;
        const double rho0 = params::rho_ps;
        //derivative follows chain rule
        //dp/drho = dp/dJ * dJ/drho

        const double J = params::rho_ps / rho;
        const double &dpdJ = K;
        const double &dJdrho = - 1.0 * rho0 / rho / rho;
        return dpdJ * dJdrho;
    }

    inline double c2(double rho, double T, bool reacted){
        const double K = params::K;
        const double rho0 = params::rho_ps;
        const double J = rho0 / rho;


        const double &dpdrho = drho(rho, T);
        const double &dpdT = dT(rho, T);

        //construct squared sound speed
        const double c2 = -dpdrho + (T / std::pow(rho, 2) / params::cv_ps) * std::pow(dpdT, 2);
        return c2;
    }
}
