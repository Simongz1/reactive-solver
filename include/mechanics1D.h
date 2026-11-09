#pragma once

inline double grad(const Eigen::VectorXd &field, unsigned int i, double dx){
    return (field[i+1] - field[i-1]) / (2.0 * dx);
}