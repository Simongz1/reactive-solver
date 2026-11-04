//initialize state

#include "modelParams.h"
#include <vector>
#include <iostream>
#include "Eigen/Sparse"
#include "Eigen/Core"
#include "computeReactionRates.h"
#include "initializeState.h"
#include <fstream>
#include <cmath>
#include <stdexcept>

//initialize old state
Eigen::VectorXd Y1_old = Eigen::VectorXd::Zero(params::N);
Eigen::VectorXd Y2_old = Eigen::VectorXd::Zero(params::N);
Eigen::VectorXd Y3_old = Eigen::VectorXd::Zero(params::N);
Eigen::VectorXd T_old = Eigen::VectorXd::Zero(params::N);

//state vector
std::vector<Eigen::VectorXd> state = {Y1_old, Y2_old, Y3_old, T_old};

//location
Eigen::VectorXd X = Eigen::VectorXd::Zero(params::N);

void initialConditions(std::vector<Eigen::VectorXd> &state)
{
    //retrieve parameters
    double Y1_init = params::Y1_init;
    double Y2_init = params::Y2_init;
    double Y3_init = params::Y3_init;
    double T_init = params::T_init;


    for (unsigned int i = 0; i < params::N; ++i)
    {
        //set T and Y1 at the specified locations
        if (i >= params::highTempStart && i <= params::highTempEnd)
        //T
        {
            state[3][i] = T_init;
            state[0][i] = Y1_init;
        }
        else
        {
            state[3][i] = params::T_0;
        }
        
        //other variables remain unchanged
        state[0][i] = Y1_init;
        state[1][i] = Y2_init;
        state[2][i] = Y3_init;
    }
}

//set the state from the initial conditions
void initializeState()
{
    state.assign(params::nvars, Eigen::VectorXd::Zero(params::N));
    X = Eigen::VectorXd::Zero(params::N);

    initialConditions(state);

    Y1_old = state[0];
    Y2_old = state[1];
    Y3_old = state[2];
    T_old = state[3];

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
             << state[0][i] << ","
             << state[1][i] << ","
             << state[2][i] << ","
             << state[3][i] << "\n";
    }
}

//set a function to assemble the residual
std::vector<Eigen::VectorXd> assembleResidual(const std::vector<Eigen::VectorXd> &state)
{   
    std::vector<Eigen::VectorXd> residual(params::nvars, Eigen::VectorXd::Zero(params::N));

    //pre-allocate rates
    Eigen::VectorXd r1(params::N);
    Eigen::VectorXd r2(params::N);

    for (unsigned int i = 0; i < params::N; ++i){
        r1[i] = reactionRates::computeArrhenius(state[3][i],
                                    params::R, 
                                    params::E1, 
                                    params::Z1);

        r2[i] = reactionRates::computeArrhenius(state[3][i],
                                    params::R, 
                                    params::E2, 
                                    params::Z2);

        //assign all residuals here 
        residual[0][i] = - r1[i] * state[0][i];
        residual[1][i] = r1[i] * state[0][i] - r2[i] * state[1][i];
        residual[2][i] = r2[i] * state[1][i];


        double Q1 = reactionRates::computeReactionHeat(params::a1, params::b1, state[3][i]);
        double Q2 = reactionRates::computeReactionHeat(params::a2, params::b2, state[3][i]);

        //set the T contribution
        residual[3][i] = (params::rho / (params::rho * params::cv)) * (Q1 * r1[i] * state[0][i] + Q2 * r2[i] * state[1][i]);

        //add conduction contribution
        if (i > 0 && i < params::N - 1){
            residual[3][i] += params::alpha / params::dx / params::dx * (state[3][i+1] - 2*state[3][i] + state[3][i-1]);
        }

        //robin BC: set temperature value and flux
        if (i == 0){
            residual[3][i] = 0.5 * params::alpha / params::dx / params::dx * (-7.0 * state[3][i] + 8.0 * state[3][i+1] - state[3][i+2]);
        }
        if (i == params::N - 1){
            residual[3][i] = 0.5 * params::alpha / params::dx / params::dx * (-7.0 * state[3][i] + 8.0 * state[3][i-1] - state[3][i-2]);
        }
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
    for (unsigned int j = 0; j < params::nvars; ++j){
        state1[j] = state_n[j] + params::dt * R[j];
    }

    //second iteration
    R = assembleResidual(state1);
    for (unsigned int j = 0; j < params::nvars; ++j){
        state2[j] = 0.75 * state_n[j] + 0.25 * state1[j] + 0.25 * params::dt * R[j];
    }

    //final iteration
    R = assembleResidual(state2);
    for (unsigned int j = 0; j < params::nvars; ++j){
        state[j] = (1.0 / 3.0) * state_n[j] + (2.0 / 3.0) * state2[j] + (2.0 / 3.0) * params::dt * R[j];
    }
}


int main(int argc, char* argv[]){

    if (argc != 9){
        std::cerr << "Usage: " << argv[0] << " N l T_init T_0 dt outputFileName numSteps stride" << std::endl;
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
        file << "step,X,Y1,Y2,Y3,T\n";
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