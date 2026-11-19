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

    //polystyrene parameters:
    const double E = 0.001;
    const double nu = 0.35;
    const double G = E / (3.0 * (1.0 - 2.0 * nu));
    const double K = 3.0 * G * (1.0 + nu) / (3 * (1.0 - 2.0 * nu));

    //shear modulus of polystyrene, G = E / (2(1 + nu)) (used by the Lipit solver)
    //note: G and K above are not the standard isotropic relations
    //(K should be E / (3(1 - 2 nu))); they are left unchanged because pressurePS uses K
    const double G_ps = E / (2.0 * (1.0 + nu));
    //polystyrene yield stress (GPa), perfectly plastic. PLACEHOLDER: set your value
    const double Y_ps = 0.05;

    //RDX shear modulus (GPa). PLACEHOLDER: set your value
    const double G_rdx = 3.0;

    //add parameters for the flyer
    const double K_flyer = 400;
    const double nu_flyer = 0.3;
    const double G_flyer = 3.0 * K_flyer * (1.0 - 2.0 * nu_flyer) / (2.0 * (1.0 + nu_flyer));

    // Multimaterial thermal parameters (J/kg-K; rho in solver units).
    const double rho_ps = 1020e-9;
    const double cv_ps = 3000;
    const double cv_air = 1000;
    const double R_air = 287;

    //plasticity parametere
    const double A_rdx = 0.2;
    const double B_rdx = 0.3;
    const double ref_ep = 1e-6;
    const double n_rdx = 0.1;
    const double m_rdx = 3;
    const double C_rdx = 0.1;
    const double Tm_rdx = 512.0;
}