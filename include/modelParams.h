#pragma once
#include <string>

namespace params{
    //domain setup
    inline int N;
    inline double l;
    inline double T_init;
    inline double T_0;
    inline double dt;

    inline double dx;
    inline unsigned int highTempStart;
    inline unsigned int highTempEnd;

    inline std::string outputFileName;
    inline unsigned int numSteps;
    inline unsigned int stride;

    inline bool use_artificial_viscosity;

    inline void update(){
        dx = l / (N - 1);
        highTempStart = 2000 * N / 5000;
        highTempEnd = 3000 * N / 5000;
    }

    //chemistry setup
    const double Y1_init = 1.0;
    const double Y2_init = 0.0;
    const double Y3_init = 1.0 - Y1_init - Y2_init;
    const int nvars = 6;
    //thermodynamic constants
    const double R = 1.987;
    const double cv = 2355.592;
    const double k = 0.361;
    const double rho = 1820e-9;
    const double alpha = k / rho / cv;
    //sakano parameters
    const double E1 = 24400;
    const double E2 = 23500;
    const double Z1 = 32103.943853583012;
    const double Z2 = 11810.38092425548;
    const double a1 = -32002.15963286241;
    const double b1 = 591.0987132187528;
    const double a2 = 7831116.710159273;
    const double b2 = -722.1193197156482;
    const double rho_init = 1820e-9;
    const double vel_init = 0.0;

    //EOS parameters
    const double A = 13398.42268091;
    const double R1 = 68.1217746;
    const double R2 = 6.81217746;
    const double B = 3810.3712782;
    const double omega = 0.37;
    //const std::string outputFileName = "solution.csv";

    //artificial viscosity parameters
    const double C0 = 0.1;
    const double C1 = 2.0;
    const double ss = std::sqrt(20.1 / rho_init);
}