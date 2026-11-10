#pragma once
#include <vector>
#include <algorithm>
#include <iostream>
#include "Eigen/Sparse"
#include "Eigen/Core"
#include "pressure.h"

namespace limiter {
    inline double computeLimit(const double & c2,
                               const double & u_left,
                               const double & u_right)
    {
        //define the limiter
        const double c = std::sqrt(c2);
        const double left = std::abs(u_left) + c;
        const double right = std::abs(u_right) + c;
        return std::max(left, right);
    }
    inline double eval(const Eigen::VectorXd & leftFlux,
                       const Eigen::VectorXd & rightFlux,
                       const Eigen::VectorXd & leftConserved,
                       const Eigen::VectorXd & rightConserved,
                       const double & c2)
    {
       //return the limited slope value
       const double average = 0.5 * (leftFlux + rightFlux);
       return average - 0.5 * computeLimit(c2, leftConserved, rightConserved);
    }
}