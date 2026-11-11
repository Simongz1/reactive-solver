//initialize state
#include "modelParams.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include "Eigen/Sparse"
#include "Eigen/Core"
#include "computeReactionRates.h"
#include "initializeState.h"
#include <fstream>
#include <cmath>
#include <stdexcept>
#include "pressure.h"
#include "mechanics1D.h"
#include "limiter.h"

//this implements conservation using a finite volume

//initialize old state
Eigen::VectorXd Y1_old = Eigen::VectorXd::Zero(params::N);
Eigen::VectorXd Y2_old = Eigen::VectorXd::Zero(params::N);
Eigen::VectorXd Y3_old = Eigen::VectorXd::Zero(params::N);
Eigen::VectorXd rho_old = Eigen::VectorXd::Zero(params::N);
Eigen::VectorXd m_old = Eigen::VectorXd::Zero(params::N);
Eigen::VectorXd rhoE_old = Eigen::VectorXd::Zero(params::N);

//state vector
std::vector<Eigen::VectorXd> state = {Y1_old, Y2_old, Y3_old, rho_old, m_old, rhoE_old};

//location
Eigen::VectorXd X = Eigen::VectorXd::Zero(params::N);

static double leftHotWidthNm = 0.0; // Zero selects the original centered hot region.

void initialConditions(std::vector<Eigen::VectorXd> &state)
{
    //retrieve parameters
    double Y1_init = params::Y1_init;
    double Y2_init = params::Y2_init;
    double Y3_init = params::Y3_init;
    double T_init = params::T_init;
    double rho_init = params::rho_init;
    double m_init = rho_init * params::vel_init;


    for (unsigned int i = 0; i < params::N; ++i)
    {
        const bool hot = leftHotWidthNm > 0.0
            ? i * params::dx < leftHotWidthNm
            : i >= params::highTempStart && i < params::highTempEnd;
        const double T = hot ? T_init : params::T_0;

        //other variables remain unchanged
        state[0][i] = rho_init * Y1_init;
        state[1][i] = rho_init * Y2_init;
        state[2][i] = rho_init * Y3_init;
        state[3][i] = rho_init;
        state[4][i] = m_init;
        state[5][i] = rho_init * params::cv * T + 0.5 * m_init * params::vel_init;
    }
}

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

double temperature(const std::vector<Eigen::VectorXd>& state, int i)
{
    const double rho = state[3][i];
    const double u = state[4][i]/rho;
    const double Y1 = state[0][i]/rho;
    const double e = state[5][i]/rho - 0.5*u*u;
    return (e - Y1*referenceEnergy(rho,false) - (1-Y1)*referenceEnergy(rho,true))/params::cv;
}

double stress(const std::vector<Eigen::VectorXd>& state, int i){
    const double rho = state[3][i];
    const double &Y1 = state[0][i] / rho;
    const double &T = temperature(state,i);
    return Y1 * pressure::value(rho, T, false) + (1.0 - Y1) * pressure::value(rho, T, true);
}

//set the state from the initial conditions
void initializeState()
{
    state.assign(state.size(), Eigen::VectorXd::Zero(params::N));
    X = Eigen::VectorXd::Zero(params::N);

    initialConditions(state);

    Y1_old = state[0];
    Y2_old = state[1];
    Y3_old = state[2];
    rho_old = state[3];
    m_old = state[4];
    rhoE_old = state[5];

    //initialize location
    for (unsigned int i = 0; i < params::N; ++i)
    {
        X[i] = i * params::dx;
    }
}

void writeToFile(int iter, const std::vector<Eigen::VectorXd>& state)
{
    std::ofstream file(params::outputFileName, std::ios::app);

    for (int i = 0; i < params::N; ++i) {
        file << iter << ","
             << X[i] << ","
             << state[0][i] / state[3][i] << ","
             << state[1][i] / state[3][i] << ","
             << state[2][i] / state[3][i] << ","
             << temperature(state,i) << ","
             << state[3][i] << ","
             << state[4][i] / state[3][i] << ","
             << state[5][i] << ","
             << stress(state,i) << "\n";
    }
}

//set a function to assemble the residual
std::vector<Eigen::VectorXd> assembleResidual(const std::vector<Eigen::VectorXd> &state)
{   
    std::vector<Eigen::VectorXd> residual(state.size(), Eigen::VectorXd::Zero(params::N));

    //pre-allocate rates
    Eigen::VectorXd r1(params::N);
    Eigen::VectorXd r2(params::N);

    // Endpoint stresses enter the gradients at the first and last interior nodes.
    Eigen::VectorXd stress(params::N), Tfield(params::N);
    // Face vector order: rhoY1, rhoY2, rhoY3, rho, m, rhoE.
    Eigen::MatrixXd conserved(6, params::N), flux(6, params::N);
    Eigen::VectorXd c2(params::N), velocity(params::N);
    for (int i = 0; i < params::N; ++i) {
        const double rho = state[3][i];
        const double T = temperature(state,i);
        Tfield[i] = T;

        //compute both pressures, then interpolate
        const double &p_unreacted = pressure::value(rho, T, false);
        const double &p_reacted = pressure::value(rho, T, true);
        const double Y1 = state[0][i] / rho;
        stress[i] = p_unreacted * Y1 + (1.0 - Y1) * p_reacted;
        const double pRho = Y1 * pressure::drho(rho,T,false) + (1-Y1) * pressure::drho(rho,T,true);
        const double stressT = Y1 * pressure::dT(rho,T,false) + (1-Y1) * pressure::dT(rho,T,true);
        c2[i] = pRho + T * stressT * stressT / (rho * rho * params::cv);
        velocity[i] = state[4][i] / rho;

        conserved.col(i) << state[0][i], state[1][i], state[2][i], rho, state[4][i], state[5][i];
        flux.col(i) = velocity[i] * conserved.col(i);
        flux(4,i) -= stress[i];
        flux(5,i) -= velocity[i] * stress[i];
    }

    // Endpoints are rigid walls with half-width control volumes.
    Eigen::MatrixXd faceFlux = Eigen::MatrixXd::Zero(6, params::N + 1);
    faceFlux(4,0) = -stress[0];
    faceFlux(4,params::N) = -stress[params::N-1];
    for (int i = 0; i+1 < params::N; ++i) {
        faceFlux.col(i+1) = limiter::eval(flux.col(i), flux.col(i+1),
            conserved.col(i), conserved.col(i+1), c2[i], c2[i+1], velocity[i], velocity[i+1]);
        faceFlux(5,i+1) -= params::k * (Tfield[i+1] - Tfield[i]) / params::dx;
        if (params::use_artificial_viscosity) {
            const double compression = std::max(0.0, velocity[i] - velocity[i+1]);
            const double q = 0.5 * (state[3][i] + state[3][i+1]) * compression
                           * (params::C0 * compression + params::C1 * params::ss);
            faceFlux(4,i+1) += q;
            faceFlux(5,i+1) += 0.5 * (velocity[i] + velocity[i+1]) * q;
        }
    }

    for (unsigned int i = 0; i < params::N; ++i){
        const double rho = state[3][i];
        r1[i] = reactionRates::computeArrhenius(Tfield[i],
                                    params::R, 
                                    params::E1, 
                                    params::Z1);

        r2[i] = reactionRates::computeArrhenius(Tfield[i],
                                    params::R, 
                                    params::E2, 
                                    params::Z2);

        /// SPECIES CONSERVATION ///

        const bool wall = i == 0 || i == params::N - 1;
        const double width = wall ? 0.5 * params::dx : params::dx;
        const Eigen::VectorXd transport = (faceFlux.col(i) - faceFlux.col(i+1)) / width;
        for (int j = 0; j < 3; ++j) residual[j][i] = transport[j];
        residual[3][i] = transport[3];
        residual[4][i] = wall ? 0.0 : transport[4];
        residual[5][i] = transport[5];

        //local sources for species
        residual[0][i] += - r1[i] * state[0][i];
        residual[1][i] += r1[i] * state[0][i] - r2[i] * state[1][i];
        residual[2][i] += r2[i] * state[1][i];

        const double Q1 = reactionRates::computeReactionHeat(params::a1, params::b1, Tfield[i]);
        const double Q2 = reactionRates::computeReactionHeat(params::a2, params::b2, Tfield[i]);

        // Heat release per reference volume, with pressure work already in the face flux.
        residual[5][i] += (params::rho_init/rho) *
            (Q1*r1[i]*state[0][i] + Q2*r2[i]*state[1][i]);
    }
    return residual;
}

void RK3Step(std::vector<Eigen::VectorXd>& state)
{
    const auto state_n = state;
    auto state1 = state_n;
    auto state2 = state_n;

    //retrieve residual
    auto R = assembleResidual(state_n);

    //first iteration
    for (unsigned int j = 0; j < state.size(); ++j){
        state1[j] = state_n[j] + params::dt * R[j];
    }

    //second iteration
    R = assembleResidual(state1);
    for (unsigned int j = 0; j < state.size(); ++j){
        state2[j] = 0.75 * state_n[j] + 0.25 * state1[j] + 0.25 * params::dt * R[j];
    }

    //final iteration
    R = assembleResidual(state2);
    for (unsigned int j = 0; j < state.size(); ++j){
        state[j] = (1.0 / 3.0) * state_n[j] + (2.0 / 3.0) * state2[j] + (2.0 / 3.0) * params::dt * R[j];
    }
}


int main(int argc, char* argv[]){

    if (argc < 9 || argc > 11){
        std::cerr << "Usage: " << argv[0] << " N l T_init T_0 dt outputFileName numSteps stride [use_artificial_viscosity: 0|1] [left_hot_width_nm]" << std::endl;
        return 1;
    }

    try {
        params::N = std::stoi(argv[1]);
        params::l = std::stod(argv[2]);
        params::T_init = std::stod(argv[3]);
        params::T_0 = std::stod(argv[4]);
        params::dt = std::stod(argv[5]);
        params::outputFileName = std::string(argv[6]);
        params::numSteps = std::stoi(argv[7]);
        params::stride = std::stoi(argv[8]);
        if (argc >= 10) {
            const std::string flag(argv[9]);
            if (flag != "0" && flag != "1")
                throw std::invalid_argument("use_artificial_viscosity must be 0 or 1");
            params::use_artificial_viscosity = flag == "1";
        }

        if (argc == 11) {
            std::size_t consumed;
            const std::string width(argv[10]);
            leftHotWidthNm = std::stod(width, &consumed);
            if (consumed != width.size() || !std::isfinite(leftHotWidthNm)
                || leftHotWidthNm <= 0 || leftHotWidthNm >= params::l)
                throw std::invalid_argument("Require 0 < left_hot_width_nm < domain length");
        }

        if (!std::isfinite(params::dt) || params::dt <= 0 || 
            params::N <= 2 || !std::isfinite(params::l) || 
            params::T_0 > params::T_init || params::l <= 0){
                throw std::invalid_argument("Fix arguments !!");
        }
    }
    
    catch (const std::exception& error){
        std::cerr << "Invalid Parameters: " << error.what() << '\n';
        return 1;
    }

    //initialization
    params::update();
    initializeState();

    //clean old file state
    {
        std::ofstream file(params::outputFileName);
        file << "step,X,Y1,Y2,Y3,T,rho,v,rhoE,p\n";
    }

    //write the initial state to a csv file
    writeToFile(1, state);

    //run one step
    RK3Step(state);
    //write the updated state to a csv file
    writeToFile(2, state);

    //iterate
    for (unsigned int iter = 0; iter < params::numSteps; ++iter){
        RK3Step(state);
        if ((iter + 1) % params::stride == 0 || iter + 1 == params::numSteps) {
            std::cout << "saving step " << iter + 1 << std::endl;
            writeToFile(iter + 3, state);
        }
    }

    return 0;
}