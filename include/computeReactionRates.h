//compute the reaction rates
#pragma once
#include <cmath>
#include "Eigen/Dense"
#include "Eigen/Sparse"

namespace reactionRates{
    inline double computeArrhenius(
        const double &temperature,
        const double &R,
        const double &E,
        const double &Z
    )
    {
        return Z * std::exp(-E / (R * temperature));
    }

    inline double computeReactionHeat(
        const double & a, const double & b,
        const double & current_T
    )
    {
        bool changing = current_T > 1736.6 ? true : false;
        if (!changing){
            return a;
        }else{
            return a + b * (current_T - 1736.6);
        }
    }
}