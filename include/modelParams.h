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

    //EOS parameters for unreacted material
    const double A_unreacted = 13398.42268091;
    const double R1_unreacted = 68.1217746;
    const double R2_unreacted = 6.81217746;
    const double B_unreacted = 3810.3712782;
    const double omega_unreacted = 0.37;

    //EOS parameters for reacted material
    const double A_reacted = 1000;
    const double R1_reacted = 59.99058313;
    const double R2_reacted = 5.999058313;
    const double B_reacted = 2738.01645291;
    const double omega_reacted = 0.77;
    //const std::string outputFileName = "solution.csv";

    //artificial viscosity parameters
    const double C0 = 0.1;
    const double C1 = 2.0;
    const double ss = std::sqrt(20.1 / rho_init);
}