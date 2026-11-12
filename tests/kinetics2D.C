// Build with the same include/library paths as the 2D solvers.
#define KINETICS2DFV_NO_MAIN
#include "../src/kinetics2DFV.C"
#include "initializeState.h"
#include <algorithm>

#define KINETICS2D_NO_MAIN
namespace fd {
#include "../src/kinetics2D.C"
}
namespace reference1d {
#include "../src/kinetics1DFV.C"
}

void near(double a, double b)
{
    if (!std::isfinite(a) || !std::isfinite(b)
        || std::abs(a-b)>2e-11*std::max({1.0,std::abs(a),std::abs(b)}))
        throw std::runtime_error("Numerical comparison failed");
}

void reduction(bool alongX)
{
    nx=alongX ? 9 : 5; ny=alongX ? 5 : 9;
    dx=2; dy=3; params::N=9; params::dx=alongX ? dx : dy; params::dt=1e-5;
    std::vector<Eigen::VectorXd> line(4,Eigen::VectorXd::Zero(9));
    for (int i=0;i<9;++i) {
        line[0][i]=0.7+0.01*i; line[1][i]=0.1; line[2][i]=0.2-0.01*i;
        line[3][i]=900+10*i*i;
    }
    State s(4,Eigen::MatrixXd::Zero(nx,ny));
    for(int j=0;j<ny;++j) for(int i=0;i<nx;++i) for(int k=0;k<4;++k)
        s[k](i,j)=line[k][alongX ? i : j];
    const auto r=assembleResidual(s);
    const auto r1=reference1d::assembleResidual(line);
    for(int j=0;j<ny;++j) for(int i=0;i<nx;++i) for(int k=0;k<4;++k)
        near(r[k](i,j),r1[k][alongX ? i : j]);
    for(int n=0;n<3;++n) { RK3Step(s); reference1d::RK3Step(line); }
    for(int j=0;j<ny;++j) for(int i=0;i<nx;++i) for(int k=0;k<4;++k)
        near(s[k](i,j),line[k][alongX ? i : j]);
}

void conductionChecks()
{
    nx=fd::nx=7; ny=fd::ny=5; dx=fd::dx=2; dy=fd::dy=3;
    State s(4,Eigen::MatrixXd::Zero(nx,ny));
    s[2].setOnes(); // Fully reacted: isolate the conduction operator.
    for(int j=0;j<ny;++j) for(int i=0;i<nx;++i)
        s[3](i,j)=800+std::pow(i*dx,2)+2*std::pow(j*dy,2);
    const auto rf=assembleResidual(s), rd=fd::assembleResidual(s);
    double weightedSum=0;
    for(int j=0;j<ny;++j) for(int i=0;i<nx;++i) {
        weightedSum+=rf[3](i,j)*dx*dy*(i==0 || i==nx-1 ? .5 : 1)*(j==0 || j==ny-1 ? .5 : 1);
        // Laplacian of x^2+2y^2 is 6 at every interior point.
        if(i>0 && i<nx-1 && j>0 && j<ny-1) {
            near(rf[3](i,j),6*params::alpha);
            near(rd[3](i,j),6*params::alpha);
        }
    }
    near(weightedSum,0);
    near(rd[3](0,0),6*params::alpha);
    State transposed;
    for(const auto& f:s) transposed.push_back(f.transpose());
    std::swap(nx,ny); std::swap(dx,dy);
    std::swap(fd::nx,fd::ny); std::swap(fd::dx,fd::dy);
    const auto rft=assembleResidual(transposed), rdt=fd::assembleResidual(transposed);
    for(int j=0;j<ny;++j) for(int i=0;i<nx;++i) {
        near(rft[3](i,j),rf[3](j,i)); near(rdt[3](i,j),rd[3](j,i));
    }
}

void reactionChecks()
{
    nx=fd::nx=7; ny=fd::ny=5; dx=fd::dx=2; dy=fd::dy=3;
    State s(4,Eigen::MatrixXd::Zero(nx,ny));
    s[0].setConstant(.7); s[1].setConstant(.1); s[2].setConstant(.2); s[3].setConstant(1800);
    const auto rf=assembleResidual(s), rd=fd::assembleResidual(s);
    const double r1=reactionRates::computeArrhenius(1800,params::R,params::E1,params::Z1);
    const double r2=reactionRates::computeArrhenius(1800,params::R,params::E2,params::Z2);
    const double heat=(reactionRates::computeReactionHeat(params::a1,params::b1,1800)*r1*.7
        +reactionRates::computeReactionHeat(params::a2,params::b2,1800)*r2*.1)/params::cv;
    for(int j=0;j<ny;++j) for(int i=0;i<nx;++i) {
        near(rf[0](i,j)+rf[1](i,j)+rf[2](i,j),0);
        for(int k=0;k<3;++k) near(rf[k](i,j),rd[k](i,j));
        near(rf[3](i,j),heat);
        near(rd[3](i,j),(i==0 || i==nx-1 || j==0 || j==ny-1) ? 0 : heat);
    }
}

int main()
{
    reduction(true); reduction(false); conductionChecks(); reactionChecks();
    std::cout << "Passed: FV 1D reduction in both directions through RK3, insulated heat conservation, FD/FV interior Laplacian, axis symmetry, reaction sources and boundary behavior.\n";
}
