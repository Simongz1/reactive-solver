// First-order finite volumes using limiter::eval in both directions.
#include <cmath>
#include <Eigen/Core>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include "modelParams.h"
#include "computeReactionRates.h"
#include "pressure.h"
#include "limiter.h"
#include "ExodusOutput.h"

using State = std::vector<Eigen::MatrixXd>;
// Fields indexed (i,j): rhoY1, rhoY2, rhoY3, rho, mx, my, rhoE.
constexpr int nvars = 7;
int nx, ny;
double dx, dy;
double leftHotWidthNm = 0.0;
double circularHotRadiusNm = 0.0; // Centered at (Lx/2, Ly/2) when positive.

// e = cv*T + e_ref(rho,Y1), with e_ref = 0 at rho_init for both phases.
double referenceEnergy(double rho, bool reacted)
{
    const double compression = 1.0 - params::rho_init/rho;
    const double omega = reacted ? params::omega_reacted : params::omega_unreacted;
    const auto coldWork = [compression](double A, double R) {
        return A*std::exp(-R)*(std::expm1(R*compression)/R - compression);
    };
    const double A = reacted ? params::A_reacted : params::A_unreacted;
    const double B = reacted ? params::B_reacted : params::B_unreacted;
    const double R1 = reacted ? params::R1_reacted : params::R1_unreacted;
    const double R2 = reacted ? params::R2_reacted : params::R2_unreacted;
    return (coldWork(A,R1) + coldWork(B,R2))/params::rho_init
         + omega*params::cv*300.0*std::log(params::rho_init/rho);
}

double temperature(const State& state, int i, int j)
{
    const double rho = state[3](i,j);
    const double u = state[4](i,j)/rho, v = state[5](i,j)/rho;
    const double Y1 = state[0](i,j)/rho;
    const double e = state[6](i,j)/rho - 0.5*(u*u + v*v);
    return (e - Y1*referenceEnergy(rho,false) - (1-Y1)*referenceEnergy(rho,true))/params::cv;
}

double stress(const State& state, int i, int j)
{
    const double rho = state[3](i,j), Y1 = state[0](i,j)/rho;
    const double T = temperature(state,i,j);
    return Y1*pressure::value(rho,T,false) + (1-Y1)*pressure::value(rho,T,true);
}

void initialConditions(State& state)
{
    const double rho = params::rho_init;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double offsetX = i*dx - 0.5*(nx-1)*dx;
            const double offsetY = j*dy - 0.5*(ny-1)*dy;
            const bool hot = circularHotRadiusNm > 0
                ? offsetX*offsetX + offsetY*offsetY <= circularHotRadiusNm*circularHotRadiusNm
                : leftHotWidthNm > 0 ? i*dx < leftHotWidthNm
                : i >= 2LL*nx/5 && i < 3LL*nx/5 && j >= 2LL*ny/5 && j < 3LL*ny/5;
            const double T = hot ? params::T_init : params::T_0;
            state[0](i,j) = rho*params::Y1_init;
            state[1](i,j) = rho*params::Y2_init;
            state[2](i,j) = rho*params::Y3_init;
            state[3](i,j) = rho;
            state[4](i,j) = rho*params::vel_init;
            state[5](i,j) = 0.0;
            state[6](i,j) = rho*params::cv*T + 0.5*rho*params::vel_init*params::vel_init;
        }
    }
}

// Evaluate the physical flux and wave speed from a cell conserved state.
struct FacePhysics {
    Eigen::VectorXd flux;
    double c2, velocity;
};

bool facePhysics(const Eigen::VectorXd& U, bool xDirection, FacePhysics& result)
{
    if (!U.allFinite() || U[3] <= 0) return false;
    const double rho = U[3], Y1 = U[0]/rho;
    // Allow only roundoff-sized species-bound violations.
    for (int k = 0; k < 3; ++k)
        if (U[k] < -1e-12*rho || U[k] > (1+1e-12)*rho) return false;
    const double u = U[4]/rho, v = U[5]/rho;
    const double e = U[6]/rho - 0.5*(u*u+v*v);
    const double T = (e - Y1*referenceEnergy(rho,false)
        - (1-Y1)*referenceEnergy(rho,true))/params::cv;
    if (!std::isfinite(T) || T <= 0) return false;
    const double sigma = Y1*pressure::value(rho,T,false) + (1-Y1)*pressure::value(rho,T,true);
    const double pRho = Y1*pressure::drho(rho,T,false) + (1-Y1)*pressure::drho(rho,T,true);
    const double stressT = Y1*pressure::dT(rho,T,false) + (1-Y1)*pressure::dT(rho,T,true);
    result.c2 = pRho + T*stressT*stressT/(rho*rho*params::cv);
    if (!std::isfinite(result.c2) || result.c2 < 0) return false;
    result.velocity = xDirection ? u : v;
    result.flux = result.velocity*U;
    result.flux[xDirection ? 4 : 5] -= sigma;
    result.flux[6] -= result.velocity*sigma;
    return result.flux.allFinite();
}

State assembleResidual(const State& state)
{
    State residual(nvars, Eigen::MatrixXd::Zero(nx,ny));
    const Eigen::Index count = Eigen::Index(nx)*ny;
    auto index = [](int i, int j) { return Eigen::Index(j)*nx+i; };
    Eigen::MatrixXd conserved(nvars,count);
    Eigen::VectorXd Tfield(count), u(count), v(count);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const auto p = index(i,j);
            const double rho = state[3](i,j);
            if (!std::isfinite(rho) || rho <= 0)
                throw std::runtime_error("Nonpositive/nonfinite density; reduce dt or check state");
            Tfield[p] = temperature(state,i,j);
            if (!std::isfinite(Tfield[p]) || Tfield[p] <= 0)
                throw std::runtime_error("Nonpositive/nonfinite temperature; reduce dt or check state");
            u[p] = state[4](i,j)/rho;
            v[p] = state[5](i,j)/rho;
            for (int k = 0; k < nvars; ++k) conserved(k,p) = state[k](i,j);
        }
    }

    // Face i+1/2 in x, or j+1/2 in y. Outer faces use i/j = -1 or n-1.
    // Piecewise-constant states: use the adjacent cells directly in limiter::eval.
    auto faceFlux = [&](int i, int j, bool xDirection) {
        const int position = xDirection ? i : j;
        const int length = xDirection ? nx : ny;
        auto sample = [&](int offset) {
            const int k = std::clamp(position+offset,0,length-1);
            return xDirection ? index(k,j) : index(i,k);
        };
        const auto left = sample(0), right = sample(1);
        const Eigen::VectorXd UL = conserved.col(left), UR = conserved.col(right);
        FacePhysics physicsL, physicsR;
        if (!facePhysics(UL,xDirection,physicsL) || !facePhysics(UR,xDirection,physicsR))
            throw std::runtime_error("Invalid cell state at flux face; reduce dt or check state");
        Eigen::VectorXd face = limiter::eval(physicsL.flux,physicsR.flux,UL,UR,
            physicsL.c2,physicsR.c2,physicsL.velocity,physicsR.velocity);
        // Retain the existing centered conduction and cell-based compression viscosity.
        // Both vanish at exterior faces, whose left and right cell are identical.
        face[6] -= params::k*(Tfield[right]-Tfield[left])/(xDirection ? dx : dy);
        if (params::use_artificial_viscosity) {
            const auto& velocity = xDirection ? u : v;
            const double compression = std::max(0.0,velocity[left]-velocity[right]);
            const double q = 0.5*(conserved(3,left)+conserved(3,right))*compression
                *(params::C0*compression + params::C1*params::ss);
            face[xDirection ? 4 : 5] += q;
            face[6] += 0.5*(velocity[left]+velocity[right])*q;
        }
        return face;
    };

    // Zero-gradient outflow: identical interior/exterior states at outer faces.
    // Endpoint control volumes have half width; corners have quarter area.
    Eigen::MatrixXd facesX(nvars,(Eigen::Index(nx)+1)*ny);
    Eigen::MatrixXd facesY(nvars,Eigen::Index(nx)*(ny+1));
    auto xface = [](int i, int j) { return Eigen::Index(j)*(nx+1)+i; };
    for (int j = 0; j < ny; ++j)
        for (int i = -1; i < nx; ++i)
            facesX.col(xface(i+1,j)) = faceFlux(i,j,true);
    for (int i = 0; i < nx; ++i)
        for (int j = -1; j < ny; ++j)
            facesY.col(index(i,j+1)) = faceFlux(i,j,false);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const bool xBoundary = i == 0 || i == nx-1;
            const bool yBoundary = j == 0 || j == ny-1;
            const double widthX = xBoundary ? 0.5*dx : dx;
            const double widthY = yBoundary ? 0.5*dy : dy;
            const Eigen::VectorXd transport =
                (facesX.col(xface(i,j))-facesX.col(xface(i+1,j)))/widthX
                + (facesY.col(index(i,j))-facesY.col(index(i,j+1)))/widthY;
            for (int k = 0; k < nvars; ++k) residual[k](i,j) = transport[k];
            // Both momentum components evolve at every boundary cell.

            const double T = Tfield[index(i,j)], rho = state[3](i,j);
            const double r1 = reactionRates::computeArrhenius(T,params::R,params::E1,params::Z1);
            const double r2 = reactionRates::computeArrhenius(T,params::R,params::E2,params::Z2);
            residual[0](i,j) -= r1*state[0](i,j);
            residual[1](i,j) += r1*state[0](i,j) - r2*state[1](i,j);
            residual[2](i,j) += r2*state[1](i,j);
            const double Q1 = reactionRates::computeReactionHeat(params::a1,params::b1,T);
            const double Q2 = reactionRates::computeReactionHeat(params::a2,params::b2,T);
            // Preserve the 1D heat source and EOS reference-energy convention.
            residual[6](i,j) += (params::rho_init/rho)
                *(Q1*r1*state[0](i,j) + Q2*r2*state[1](i,j));
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


// Derived fields preserve the 1D output convention: p is Cauchy stress,
// i.e. negative gauge pressure, as returned by pressure::value.
State outputFields(const State& state)
{
    State fields(9,Eigen::MatrixXd::Zero(nx,ny));
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double rho = state[3](i,j);
            for (int k = 0; k < 3; ++k) fields[k](i,j) = state[k](i,j)/rho;
            fields[3](i,j) = temperature(state,i,j);
            fields[4](i,j) = rho;
            fields[5](i,j) = state[4](i,j)/rho;
            fields[6](i,j) = state[5](i,j)/rho;
            fields[7](i,j) = state[6](i,j);
            fields[8](i,j) = stress(state,i,j);
        }
    }
    return fields;
}

void writeToFile(std::ostream& file, long long step, const State& fields)
{
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            file << step << "," << i*dx << "," << j*dy;
            for (const auto& field : fields) file << "," << field(i,j);
            file << "\n";
        }
    }
}

#ifndef MECHANICS2DFV_LIMITER_NO_MAIN
int main(int argc, char* argv[])
{
    if (argc < 11 || argc > 14) {
        std::cerr << "Usage: " << argv[0]
            << " Nx Ny Lx Ly T_init T_0 dt output.e|output.csv numSteps stride [viscosity: 0|1] [left_hot_width_nm | --circular [radius_nm]]\n";
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
            throw std::invalid_argument("Expected --circular before radius_nm");
        } else if (argc == 13) {
            std::size_t consumed;
            const std::string width(argv[12]);
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
            file << std::setprecision(17) << "step,X,Y,Y1,Y2,Y3,T,rho,vx,vy,rhoE,p\n";
        } else if (extension == ".e" || extension == ".exo" || extension == ".exodus") {
            std::vector<std::string> names = {"Y1","Y2","Y3","T","rho","vx","vy","rhoE","p"};
            exodus = std::make_unique<ExodusOutput>(argv[8],nx,ny,dx,dy,names);
        } else {
            throw std::invalid_argument("Use .e, .exo, .exodus, or .csv");
        }
        auto save = [&](long long step) {
            const State fields = outputFields(state);
            if (exodus) {
                exodus->write((step-1)*params::dt,fields);
            } else {
                writeToFile(file,step,fields);
            }
        };
        // Match the snapshot numbering and initial RK step in mechanics1DFV.C.
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
