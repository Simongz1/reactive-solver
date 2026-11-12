// Finite-difference mechanics with SSP RK3.
#include <cmath>
#include "ExodusOutput.h"
#include <filesystem>
#include <memory>
#include "modelParams.h"
#include "computeReactionRates.h"
#include "pressure.h"
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using State = std::vector<Eigen::MatrixXd>;
// Fields are indexed (i, j): Y1, Y2, Y3, T, rho, mx, my.
constexpr int nvars = 7;
int nx, ny;
double dx, dy;
double leftHotWidthNm = 0.0;
double circularHotRadiusNm = 0.0; // Centered at (Lx/2, Ly/2) when positive.

double shockSpeed = 0.0;
double pistonSpeed = 0.0;

// Unreacted Hugoniot from 300 K and zero gauge pressure.
// eta = 1 - rho0/rho. The Rayleigh line gives p = rho0*D^2*eta.
struct ShockState {
    double density, velocity, temperature, pressure;
};

ShockState downstreamState(double speed)
{
    const double rho0 = params::rho_init, cv = params::cv;
    const double omega = params::omega_unreacted;
    const double r1 = params::R1_unreacted, r2 = params::R2_unreacted;
    const double a = params::A_unreacted*std::exp(-r1);
    const double b = params::B_unreacted*std::exp(-r2);
    const double slope = rho0*speed*speed;
    const double acousticSlope = a*r1 + b*r2 + rho0*cv*omega*omega*300;
    if (!std::isfinite(slope) || slope <= acousticSlope)
        throw std::invalid_argument("Shock speed must exceed the upstream adiabatic sound speed");

    // With constant cv, de/dV at fixed T is T*dp/dT - p.
    // Integrate the cold pressure and thermal reference term, then impose
    // e2-e1 = (p2+p1)*(V1-V2)/2 to obtain the temperature rise.
    auto temperatureRise = [&](double eta) {
        const double coldWork = a*(std::expm1(r1*eta)/r1 - eta)
                              + b*(std::expm1(r2*eta)/r2 - eta);
        return (0.5*slope*eta*eta - coldWork)/(rho0*cv)
             - omega*300*std::log1p(-eta);
    };
    auto pressureMismatch = [&](double eta) {
        const double cold = a*std::expm1(r1*eta) + b*std::expm1(r2*eta);
        return cold + omega*rho0*cv*temperatureRise(eta)/(1-eta) - slope*eta;
    };
    // The D-squared terms cancel at the strong-shock compression limit.
    double low = 0, high = 2/(2+omega);
    if (!(pressureMismatch(high) > 0))
        throw std::invalid_argument("No finite compressive shock state could be bracketed");
    for (int n = 0; n < 80; ++n) {
        const double mid = (low+high)/2;
        if (pressureMismatch(mid) < 0) low = mid;
        else high = mid;
    }
    const double eta = (low+high)/2;
    const double rho = rho0/(1-eta);
    const double temperature = 300 + temperatureRise(eta);
    const double pressureBehind = -pressure::value(rho,temperature,false);
    if (!(temperature >= 300 && std::isfinite(temperature) && std::isfinite(pressureBehind)))
        throw std::invalid_argument("Shock solution has invalid temperature or pressure");
    return {rho,speed*eta,temperature,pressureBehind};
}

void initialConditions(State& state)
{
    state[0].setConstant(params::Y1_init);
    state[1].setConstant(params::Y2_init);
    state[2].setConstant(params::Y3_init);
    state[4].setConstant(params::rho_init);
    if (pistonSpeed > 0) {
        state[3].setConstant(params::T_0);
        state[5].row(0) = pistonSpeed*state[4].row(0);
        return;
    }
    if (shockSpeed > 0) {
        const auto behind = downstreamState(shockSpeed);
        const int leftNodes = (nx-2)/10 + 1;
        state[3].setConstant(300.0);
        state[3].topRows(leftNodes).setConstant(behind.temperature);
        state[4].topRows(leftNodes).setConstant(behind.density);
        state[5].topRows(leftNodes).setConstant(behind.density*behind.velocity);
        std::cout.precision(12);
        std::cout << "Initial shock speed " << shockSpeed << " nm/ns, particle speed "
                  << behind.velocity << " nm/ns, density ratio " << behind.density/params::rho_init
                  << ", T " << behind.temperature << " K, pressure " << behind.pressure << " GPa\n";
        return;
    }
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

double artificialPressure(double rhoL, double rhoR, double mL, double mR)
{
    const double compression = std::max(0.0, mL/rhoL - mR/rhoR);
    return 0.5*(rhoL+rhoR)*compression*(params::C0*compression + params::C1*params::ss);
}

State assembleResidual(const State& state)
{
    State residual(nvars, Eigen::MatrixXd::Zero(nx,ny));
    const auto& T = state[3];
    const auto& density = state[4];
    const auto& mx = state[5];
    const auto& my = state[6];
    Eigen::MatrixXd stress(nx,ny), facePressureX, facePressureY;
    if (params::use_artificial_viscosity) {
        facePressureX.resize(nx-1,ny);
        facePressureY.resize(nx,ny-1);
    }
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double rho = density(i,j), Y1 = state[0](i,j);
            stress(i,j) = Y1 * pressure::value(rho,T(i,j),false)
                + (1-Y1) * pressure::value(rho,T(i,j),true);
            if (params::use_artificial_viscosity) {
                if (i+1 < nx)
                    facePressureX(i,j) = artificialPressure(rho,density(i+1,j),mx(i,j),mx(i+1,j));
                if (j+1 < ny)
                    facePressureY(i,j) = artificialPressure(rho,density(i,j+1),my(i,j),my(i,j+1));
            }
        }
    }

    for (int j = 0; j < ny; ++j) {
        const int bottom = std::max(0,j-1), top = std::min(ny-1,j+1);
        const bool yWall = j == 0 || j == ny-1;
        for (int i = 0; i < nx; ++i) {
            const int left = std::max(0,i-1), right = std::min(nx-1,i+1);
            const bool xWall = i == 0 || i == nx-1;
            auto gradX = [&](const Eigen::MatrixXd& f) {
                return (f(right,j)-f(left,j))/((right-left)*dx);
            };
            auto gradY = [&](const Eigen::MatrixXd& f) {
                return (f(i,top)-f(i,bottom))/((top-bottom)*dy);
            };
            const double rho = density(i,j), temp = T(i,j);
            const double u = mx(i,j)/rho, v = my(i,j)/rho;
            const double Y1 = state[0](i,j), Y2 = state[1](i,j);
            const double r1 = reactionRates::computeArrhenius(temp,params::R,params::E1,params::Z1);
            const double r2 = reactionRates::computeArrhenius(temp,params::R,params::E2,params::Z2);
            residual[0](i,j) = -r1*Y1;
            residual[1](i,j) = r1*Y1 - r2*Y2;
            residual[2](i,j) = r2*Y2;

            const double divM = gradX(mx) + gradY(my);
            // In shock mode the left boundary supplies the post-shock density
            // and x momentum. The right boundary remains a rigid wall.
            residual[4](i,j) = (shockSpeed > 0 && i == 0) ? 0.0 : -divM;
            // Expand div(m tensor m / rho) with the same product rule as in 1D.
            // Only the normal momentum is fixed at a rigid wall.
            // At the driven boundary mx = rho * pistonSpeed at every RK stage.
            if (i == 0 && pistonSpeed > 0)
                residual[5](i,j) = pistonSpeed*residual[4](i,j);
            if (!xWall) {
                residual[5](i,j) = -u*divM - u*gradX(mx) - v*gradY(mx)
                    + u*(u*gradX(density) + v*gradY(density)) + gradX(stress);
                if (params::use_artificial_viscosity)
                    residual[5](i,j) -= (facePressureX(i,j)-facePressureX(i-1,j))/dx;
            }
            if (!yWall) {
                residual[6](i,j) = -v*divM - u*gradX(my) - v*gradY(my)
                    + v*(u*gradX(density) + v*gradY(density)) + gradY(stress);
                if (params::use_artificial_viscosity)
                    residual[6](i,j) -= (facePressureY(i,j)-facePressureY(i,j-1))/dy;
            }

            residual[3](i,j) = conduction(T,i,j,1,0,dx) + conduction(T,i,j,0,1,dy);
            // The 1D endpoint temperature equation contains conduction only.
            if (!xWall && !yWall) {
                const double Q1 = reactionRates::computeReactionHeat(params::a1,params::b1,temp);
                const double Q2 = reactionRates::computeReactionHeat(params::a2,params::b2,temp);
                const double J = params::rho_init/rho;
                const double Jdot = J*(divM - u*gradX(density) - v*gradY(density))/rho;
                const double dStressdT = Y1*pressure::dT(rho,temp,false)
                    + (1-Y1)*pressure::dT(rho,temp,true);
                residual[3](i,j) += J/params::cv*(Q1*r1*Y1 + Q2*r2*Y2)
                    + dStressdT/(rho*params::cv)*Jdot/J;
            }
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
            for (int k = 0; k <= 4; ++k) file << "," << state[k](i,j);
            file << "," << state[5](i,j)/state[4](i,j)
                 << "," << state[6](i,j)/state[4](i,j) << "\n";
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc < 11 || argc > 14) {
        std::cerr << "Usage: " << argv[0]
            << " Nx Ny Lx Ly T_init T_0 dt output.e|output.csv numSteps stride [viscosity: 0|1] [left_hot_width_nm | --circular [radius_nm] | --shock-speed D_nm_per_ns | --piston-speed u_nm_per_ns]\n";
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
        if (argc >= 12) {
            const std::string flag(argv[11]);
            if (flag != "0" && flag != "1") throw std::invalid_argument("Viscosity must be 0 or 1");
            params::use_artificial_viscosity = flag == "1";
        }
        if (argc >= 13 && std::string(argv[12]) == "--circular") {
            circularHotRadiusNm = 0.1*lx; // Match solver2D.C's default circle.
            if (argc == 14) {
                std::size_t consumed;
                const std::string radius(argv[13]);
                circularHotRadiusNm = std::stod(radius,&consumed);
                if (consumed != radius.size())
                    throw std::invalid_argument("Invalid circular hotspot radius");
            }
            if (!std::isfinite(circularHotRadiusNm) || circularHotRadiusNm <= 0)
                throw std::invalid_argument("Circular hotspot radius must be finite and positive");
        } else if (argc == 14) {
            const std::string mode(argv[12]);
            const std::string speed(argv[13]);
            std::size_t used;
            const double value = std::stod(speed, &used);
            if (used != speed.size() || !std::isfinite(value) || value <= 0)
                throw std::invalid_argument("Speed must be finite and positive");
            if (mode == "--piston-speed") pistonSpeed = value;
            else if (mode == "--shock-speed") shockSpeed = value;
            else throw std::invalid_argument("Expected --piston-speed or --shock-speed");
        }
        else if (argc == 13) {
            std::size_t consumed;
            const std::string width(argv[12]);
            leftHotWidthNm = std::stod(width,&consumed);
            if (consumed != width.size() || !std::isfinite(leftHotWidthNm)
                || !(leftHotWidthNm > 0 && leftHotWidthNm < lx))
                throw std::invalid_argument("Hot width must be between zero and Lx");
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
            file << "step,X,Y,Y1,Y2,Y3,T,rho,vx,vy\n";
        } else if (extension == ".e" || extension == ".exo" || extension == ".exodus") {
            std::vector<std::string> names = {"Y1","Y2","Y3","T","rho","vx","vy"};
            exodus = std::make_unique<ExodusOutput>(argv[8],nx,ny,dx,dy,names);
        } else {
            throw std::invalid_argument("Use .e, .exo, .exodus, or .csv");
        }
        auto save = [&](long long step) {
            if (exodus) {
                State fields = state;
                fields[5].array() /= state[4].array();
                fields[6].array() /= state[4].array();
                exodus->write((step-1)*params::dt,fields);
            } else {
                writeToFile(file,step,state);
            }
        };
        // Match the snapshot numbering and initial RK step in mechanics1D.C.
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
