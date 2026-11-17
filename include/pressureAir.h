#pragma once
#include <cmath>
#include "modelParams.h"

namespace pressureAir{

    inline double value(double rho, double T){
        //compute the ideal gas pressure for air
        const double R = params::R_air;
        return rho * R * T;
    }

    inline double dT(double rho, double T){
        //computes the derivative of pressure with respect to temperature
        const double R = params::R_air;
        return rho * R;
    }

    inline double drho(double rho, double T){
        //compute the derivative of pressure with respect to density
        const double R = params::R_air;
        return R * T;
    }

    inline double c2(double rho, double T){
        //computes the square sound speed
        
        const double &dpdrho = drho(rho, T);
        const double &dpdT = dT(rho, T);

        //construct squared sound speed
        const double c2 = dpdrho + (T / std::pow(rho, 2) / params::cv_air) * std::pow(dpdT, 2);
        return c2;
    }
}