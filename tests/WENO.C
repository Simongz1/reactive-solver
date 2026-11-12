#include "WENO.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

void near(double a, double b) {
    if (!std::isfinite(a) || std::abs(a-b)>2e-12*std::max({1.0,std::abs(a),std::abs(b)}))
        throw std::runtime_error("WENO polynomial/symmetry check failed");
}

int main() {
    // Exact finite-volume averages of monomials on cells centered at -2,...,3.
    for (int degree=0;degree<=4;++degree) {
        double q[6];
        for(int k=0;k<6;++k) {
            const double center=k-2;
            q[k]=(std::pow(center+.5,degree+1)-std::pow(center-.5,degree+1))/(degree+1);
        }
        const auto result=WENO::reconstruct(q[0],q[1],q[2],q[3],q[4],q[5]);
        if(degree<=2) { near(result.left,std::pow(.5,degree)); near(result.right,std::pow(.5,degree)); }
        if(degree<=2) {
            for(double candidate:WENO::leftCandidates(q[0],q[1],q[2],q[3],q[4])) near(candidate,std::pow(.5,degree));
            for(double candidate:WENO::rightCandidates(q[1],q[2],q[3],q[4],q[5])) near(candidate,std::pow(.5,degree));
        }
        const auto reversed=WENO::reconstruct(q[5],q[4],q[3],q[2],q[1],q[0]);
        near(reversed.left,result.right); near(reversed.right,result.left);
        const auto boundary=WENO::reconstruct(q[0],q[1],q[2],q[3],q[4],q[5],false);
        near(boundary.left,q[2]); near(boundary.right,q[3]);
    }
    double previousL=0, previousR=0;
    for(double h:{.2,.1,.05}) {
        double q[6]; const double face=.27;
        for(int k=0;k<6;++k) {
            const double center=face+(k-2.5)*h;
            q[k]=(std::cos(center-.5*h)-std::cos(center+.5*h))/h;
        }
        const auto result=WENO::reconstruct(q[0],q[1],q[2],q[3],q[4],q[5]);
        const double errorL=std::abs(result.left-std::sin(face));
        const double errorR=std::abs(result.right-std::sin(face));
        if(previousL && (previousL/errorL<25 || previousR/errorR<25))
            throw std::runtime_error("Expected fifth-order face reconstruction");
        previousL=errorL; previousR=errorR;
    }
    const auto weights=WENO::nonlinearWeights(WENO::smoothness(0,0,0,1,1));
    if(weights[0]<.999) throw std::runtime_error("Discontinuous stencils were not suppressed");
    const auto jump=WENO::reconstruct(0,0,0,1,1,1);
    if(std::abs(jump.left)>1e-10 || std::abs(jump.right-1)>1e-10)
        throw std::runtime_error("Step reconstruction overshoot");
    const auto constantWeights=WENO::nonlinearWeights({{0,0,0}});
    near(constantWeights[0],.1); near(constantWeights[1],.6); near(constantWeights[2],.3);
    std::array<std::array<double,6>,4> system;
    for(int i=0;i<6;++i) {
        system[0][i]=(i<3 ? .8 : .1)*1e-6;
        system[1][i]=(.1+.01*i)*1e-6;
        system[2][i]=(.05+.02*i)*1e-6;
        system[3][i]=system[0][i]+system[1][i]+system[2][i];
    }
    const auto face=WENO::reconstructSystem(system);
    near((face[0].left+face[1].left+face[2].left)/face[3].left,1);
    near((face[0].right+face[1].right+face[2].right)/face[3].right,1);
    for(auto& field:system) for(auto& q:field) q*=1e10;
    const auto scaled=WENO::reconstructSystem(system);
    for(int k=0;k<4;++k) {
        near(scaled[k].left/1e10,face[k].left);
        near(scaled[k].right/1e10,face[k].right);
    }
    // Rotating a momentum vector must not change the scalar reconstruction
    // or the common nonlinear weights. Independent component scaling fails this.
    std::array<std::array<double,6>,3> vectorStencil;
    for(int i=0;i<6;++i) {
        vectorStencil[0][i]=.2+i*.03;
        vectorStencil[1][i]=.001*(i-2)*(i-2)-.002;
        vectorStencil[2][i]=1+.002*i;
    }
    const auto unrotated=WENO::reconstructSystem(vectorStencil,true,0);
    const double cosine=std::cos(.71), sine=std::sin(.71);
    for(int i=0;i<6;++i) {
        const double x=vectorStencil[0][i], y=vectorStencil[1][i];
        vectorStencil[0][i]=cosine*x-sine*y;
        vectorStencil[1][i]=sine*x+cosine*y;
    }
    const auto rotated=WENO::reconstructSystem(vectorStencil,true,0);
    near(rotated[0].left,cosine*unrotated[0].left-sine*unrotated[1].left);
    near(rotated[1].left,sine*unrotated[0].left+cosine*unrotated[1].left);
    near(rotated[0].right,cosine*unrotated[0].right-sine*unrotated[1].right);
    near(rotated[1].right,sine*unrotated[0].right+cosine*unrotated[1].right);
    near(rotated[2].left,unrotated[2].left); near(rotated[2].right,unrotated[2].right);
    std::cout << "Passed: candidate polynomials, quadratic exactness, mirrored states, boundary closure, fifth-order smooth reconstruction, nonlinear stencil suppression, species identity and scale invariance.\n";
}
