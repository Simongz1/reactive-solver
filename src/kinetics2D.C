// Finite-difference reaction-diffusion solver, following mechanics2D.C.
#include <cmath>
#include <Eigen/Core>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include "modelParams.h"
#include "computeReactionRates.h"
#include "ExodusOutput.h"

using State = std::vector<Eigen::MatrixXd>;
// Fields indexed (i,j): Y1, Y2, Y3, T.
constexpr int nvars = 4;
int nx, ny;
double dx, dy;
double leftHotWidthNm = 0.0;
double circularHotRadiusNm = 0.0; // Centered at (Lx/2, Ly/2) when positive.

void initialConditions(State& state)
{
    state[0].setConstant(params::Y1_init);
    state[1].setConstant(params::Y2_init);
    state[2].setConstant(params::Y3_init);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double offsetX = i*dx - 0.5*(nx-1)*dx;
            const double offsetY = j*dy - 0.5*(ny-1)*dy;
            const bool hot = circularHotRadiusNm > 0
                ? offsetX*offsetX + offsetY*offsetY <= circularHotRadiusNm*circularHotRadiusNm
                : leftHotWidthNm > 0 ? i*dx < leftHotWidthNm
                : i >= 2LL*nx/5 && i < 3LL*nx/5 && j >= 2LL*ny/5 && j < 3LL*ny/5;
            state[3](i,j) = hot ? params::T_init : params::T_0;
        }
    }
}

// The 1D centered conduction stencil and its one-sided endpoint closure.
double conduction(const Eigen::MatrixXd& T, int i, int j, int di, int dj, double h)
{
    const int k = di ? i : j;
    const int n = di ? nx : ny;
    if (k == 0 || k == n-1) {
        const int sign = k == 0 ? 1 : -1;
        return 0.5 * params::alpha / (h*h) * (-7*T(i,j)
            + 8*T(i+sign*di,j+sign*dj) - T(i+2*sign*di,j+2*sign*dj));
    }
    return params::alpha / (h*h) * (T(i+di,j+dj) - 2*T(i,j) + T(i-di,j-dj));
}

State assembleResidual(const State& state)
{
    State residual(nvars,Eigen::MatrixXd::Zero(nx,ny));
    const auto& T = state[3];

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double temp = T(i,j);
            const double Y1 = state[0](i,j), Y2 = state[1](i,j);
            const double r1 = reactionRates::computeArrhenius(temp,params::R,params::E1,params::Z1);
            const double r2 = reactionRates::computeArrhenius(temp,params::R,params::E2,params::Z2);

            //local sources for species
            residual[0](i,j) = -r1*Y1;
            residual[1](i,j) = r1*Y1 - r2*Y2;
            residual[2](i,j) = r2*Y2;

            const double Q1 = reactionRates::computeReactionHeat(params::a1,params::b1,temp);
            const double Q2 = reactionRates::computeReactionHeat(params::a2,params::b2,temp);
            const bool xWall = i == 0 || i == nx-1;
            const bool yWall = j == 0 || j == ny-1;

            // Same one-sided boundary closure and interior heating as mechanics2D.C.
            residual[3](i,j) = conduction(T,i,j,1,0,dx) + conduction(T,i,j,0,1,dy);
            if (!xWall && !yWall)
                residual[3](i,j) += (Q1*r1*Y1 + Q2*r2*Y2)/params::cv;
        }
    }
    return residual;
}

void RK3Step(State& state)
{
    const auto state_n = state;
    auto state1 = state_n;
    auto state2 = state_n;

    //retrieve residual
    auto R = assembleResidual(state_n);

    //first iteration
    for (unsigned int j = 0; j < nvars; ++j){
        state1[j] = state_n[j] + params::dt * R[j];
    }

    //second iteration
    R = assembleResidual(state1);
    for (unsigned int j = 0; j < nvars; ++j){
        state2[j] = 0.75 * state_n[j] + 0.25 * state1[j] + 0.25 * params::dt * R[j];
    }

    //final iteration
    R = assembleResidual(state2);
    for (unsigned int j = 0; j < nvars; ++j){
        state[j] = (1.0 / 3.0) * state_n[j] + (2.0 / 3.0) * state2[j] + (2.0 / 3.0) * params::dt * R[j];
    }
}


void writeToFile(std::ostream& file, long long step, const State& state)
{
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            file << step << "," << i*dx << "," << j*dy;
            for (int k = 0; k < nvars; ++k) file << "," << state[k](i,j);
            file << "\n";
        }
    }
}

#ifndef KINETICS2D_NO_MAIN
int main(int argc, char* argv[])
{
    if (argc < 11 || argc > 13) {
        std::cerr << "Usage: " << argv[0]
            << " Nx Ny Lx Ly T_init T_0 dt output.e|output.csv numSteps stride [left_hot_width_nm | --circular [radius_nm]]\n";
        return 1;
    }
    try {
        nx = std::stoi(argv[1]);
        ny = std::stoi(argv[2]);
        const double lx = std::stod(argv[3]), ly = std::stod(argv[4]);
        params::T_init = std::stod(argv[5]);
        params::T_0 = std::stod(argv[6]);
        params::dt = std::stod(argv[7]);
        const int steps = std::stoi(argv[9]), stride = std::stoi(argv[10]);
        if (nx < 3 || ny < 3 || !(lx > 0 && ly > 0 && params::dt > 0)
            || !std::isfinite(lx) || !std::isfinite(ly) || !std::isfinite(params::dt)
            || !(params::T_init >= params::T_0 && params::T_0 > 0)
            || !std::isfinite(params::T_init) || steps < 0 || stride < 1)
            throw std::invalid_argument("Invalid grid, temperatures, time step or output interval");
        if (argc >= 12 && std::string(argv[11]) == "--circular") {
            circularHotRadiusNm = 0.1*lx; // Match solver2D.C's default circle.
            if (argc == 13) {
                std::size_t consumed;
                const std::string radius(argv[12]);
                circularHotRadiusNm = std::stod(radius,&consumed);
                if (consumed != radius.size())
                    throw std::invalid_argument("Invalid circular hotspot radius");
            }
            if (!std::isfinite(circularHotRadiusNm) || circularHotRadiusNm <= 0)
                throw std::invalid_argument("Circular hotspot radius must be finite and positive");
        } else if (argc == 13) {
            throw std::invalid_argument("Expected --circular before radius_nm");
        } else if (argc == 12) {
            std::size_t consumed;
            const std::string width(argv[11]);
            leftHotWidthNm = std::stod(width,&consumed);
            if (consumed != width.size() || !std::isfinite(leftHotWidthNm)
                || leftHotWidthNm <= 0 || leftHotWidthNm >= lx)
                throw std::invalid_argument("Require 0 < left_hot_width_nm < Lx");
        }
        dx = lx/(nx-1);
        dy = ly/(ny-1);
        State state(nvars,Eigen::MatrixXd::Zero(nx,ny));
        initialConditions(state);
        std::ofstream file;
        std::unique_ptr<ExodusOutput> exodus;
        const auto extension = std::filesystem::path(argv[8]).extension();
        if (extension == ".csv") {
            file.exceptions(std::ios::failbit | std::ios::badbit);
            file.open(argv[8]);
            file << std::setprecision(17) << "step,X,Y,Y1,Y2,Y3,T\n";
        } else if (extension == ".e" || extension == ".exo" || extension == ".exodus") {
            std::vector<std::string> names = {"Y1","Y2","Y3","T"};
            exodus = std::make_unique<ExodusOutput>(argv[8],nx,ny,dx,dy,names);
        } else {
            throw std::invalid_argument("Use .e, .exo, .exodus, or .csv");
        }
        auto save = [&](long long step) {
            if (exodus) {
                exodus->write((step-1)*params::dt,state);
            } else {
                writeToFile(file,step,state);
            }
        };
        // Match the snapshot numbering and initial RK step in kinetics1DFV.C.
        save(1);
        RK3Step(state);
        save(2);
        for (long long iter = 1; iter <= steps; ++iter) {
            RK3Step(state);
            if (iter % stride == 0 || iter == steps){
                save(iter+2);
                std::cout << "Saved step " << iter << "\n";
            }    
        }
        if (exodus) exodus->close();
        else file.close();
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}

#endif
