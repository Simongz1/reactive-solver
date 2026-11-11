#pragma once
#include <vector>
#include <algorithm>
#include <iostream>
#include "Eigen/Sparse"
#include "Eigen/Core"
#include "pressure.h"

namespace limiter {

    inline Eigen::VectorXd eval(const Eigen::VectorXd & leftFlux,
                       const Eigen::VectorXd & rightFlux,
                       const Eigen::VectorXd & leftConserved,
                       const Eigen::VectorXd & rightConserved,
                       const double & c2_left,
                       const double & c2_right,
                       const double & leftVelocity, 
                       const double & rightVelocity)
    {
       //left and right sound speeds
       const double c_left = std::sqrt(c2_left);
       const double c_right = std::sqrt(c2_right);

       //compute the limit at the interface
       const double limitFace = std::max(std::abs(leftVelocity) + c_left, std::abs(rightVelocity) + c_right);
       return 0.5 * (rightFlux + leftFlux) - 0.5 * limitFace * (rightConserved - leftConserved);
    }
}