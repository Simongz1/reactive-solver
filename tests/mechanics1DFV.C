// Compile twice: default tests WENO; -DTEST_LIMITER tests original limiter.
#ifdef TEST_LIMITER
#define MECHANICS2DFV_LIMITER_NO_MAIN
#include "../src/mechanics2DFV_limiter.C"
#else
#define MECHANICS2DFV_NO_MAIN
#include "../src/mechanics2DFV.C"
#endif
#include "initializeState.h"
#include "mechanics1D.h"

namespace one {
#ifdef TEST_LIMITER
#define MECHANICS1DFV_LIMITER_NO_MAIN
#include "../src/mechanics1DFV_limiter.C"
#else
#define MECHANICS1DFV_WENO_NO_MAIN
#include "../src/mechanics1DFV_WENO.C"
#endif
}

void near(double a,double b) {
    if(!std::isfinite(a) || !std::isfinite(b)
       || std::abs(a-b)>1e-10*std::max({1.0,std::abs(a),std::abs(b)}))
        throw std::runtime_error("1D/2D mismatch");
}

void reduction(bool alongX)
{
    nx=alongX ? 17 : 9; ny=alongX ? 9 : 17;
    dx=2; dy=3;
    params::N=17; params::dx=alongX ? dx : dy; params::dt=1e-6;
    std::vector<Eigen::VectorXd> line(6,Eigen::VectorXd::Zero(17));
    for(int i=0;i<17;++i) {
        const double rho=params::rho_init*(1+.01*std::sin(.3*i));
        const double Y1=.7+.05*std::sin(.2*i), Y2=.1;
        const double v=.1*(i-8), T=800+15*i;
        line[0][i]=rho*Y1; line[1][i]=rho*Y2; line[2][i]=rho*(1-Y1-Y2);
        line[3][i]=rho; line[4][i]=rho*v;
        line[5][i]=rho*(params::cv*T+Y1*one::referenceEnergy(rho,false)
            +(1-Y1)*one::referenceEnergy(rho,true)+.5*v*v);
    }
    State s(nvars,Eigen::MatrixXd::Zero(nx,ny));
    const int mapping[]={0,1,2,3,alongX ? 4 : 5,6};
    for(int j=0;j<ny;++j) for(int i=0;i<nx;++i) for(int k=0;k<6;++k)
        s[mapping[k]](i,j)=line[k][alongX ? i : j];
    auto r=assembleResidual(s); auto r1=one::assembleResidual(line);
    for(int j=0;j<ny;++j) for(int i=0;i<nx;++i) for(int k=0;k<6;++k)
        near(r[mapping[k]](i,j),r1[k][alongX ? i : j]);
    for(int iteration=0;iteration<3;++iteration) { RK3Step(s); one::RK3Step(line); }
    for(int j=0;j<ny;++j) for(int i=0;i<nx;++i) {
        for(int k=0;k<6;++k) near(s[mapping[k]](i,j),line[k][alongX ? i : j]);
        near(s[alongX ? 5 : 4](i,j),0);
    }
}
int main() {
    for(bool viscosity:{false,true}) {
        params::use_artificial_viscosity=viscosity;
        reduction(true); reduction(false);
    }
    std::cout << "Passed: 1D/2D residual and RK3 agreement in both directions, all boundary nodes, viscosity off/on.\n";
}
