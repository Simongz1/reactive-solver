#pragma once
#ifndef FVUTILS_H
#define FVUTILS_H

#include "Eigen/Core"
#include <cmath>
#include "Eigen/Sparse"
#include "Eigen/Dense"

namespace FVUtils{
    inline struct face{
        std::vector<double> normal;
    };
}

#endif // FVUTILS_H