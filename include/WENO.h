#pragma once
#include <array>
#include <algorithm>
#include <cmath>

// WENO-JS5: finite-volume candidates with nonlinear Jiang-Shu weights.
// Inputs are cell averages on a uniform grid, not nodal point samples.
// Smoothness is evaluated after scaling by the stencil magnitude.
// For a conserved system, use common weights to preserve linear field identities.
namespace WENO {

    inline std::array<double,3> leftCandidates(double im2, double im1,
                                              double i, double ip1, double ip2)
    {
        return {{(2*im2 - 7*im1 + 11*i)/6,
                 (-im1 + 5*i + 2*ip1)/6,
                 (2*i + 5*ip1 - ip2)/6}};
    }

    inline std::array<double,3> rightCandidates(double im1, double i,
                                               double ip1, double ip2, double ip3)
    {
        // Mirror the left candidates about the face i+1/2.
        return {{(11*ip1 - 7*ip2 + 2*ip3)/6,
                 (-ip2 + 5*ip1 + 2*i)/6,
                 (2*ip1 + 5*i - im1)/6}};
    }

    // Inputs ordered from the upwind side: i-2,...,i+2 for a left trace.
    // Reverse the stencil for a right trace to use exactly the same formulas.
    inline std::array<double,3> smoothnessWithScale(double a, double b, double c, double d, double e,
                                                   double scale)
    {
        if (scale == 0) return {{0,0,0}};
        a/=scale; b/=scale; c/=scale; d/=scale; e/=scale;
        auto square = [](double x) { return x*x; };
        return {{13.0/12*square(a-2*b+c) + 0.25*square(a-4*b+3*c),
                 13.0/12*square(b-2*c+d) + 0.25*square(b-d),
                 13.0/12*square(c-2*d+e) + 0.25*square(3*c-4*d+e)}};
    }

    inline std::array<double,3> smoothness(double a, double b, double c, double d, double e)
    {
        const double scale=std::max({std::abs(a),std::abs(b),std::abs(c),std::abs(d),std::abs(e)});
        return smoothnessWithScale(a,b,c,d,e,scale);
    }

    inline std::array<double,3> nonlinearWeights(const std::array<double,3>& beta)
    {
        constexpr double epsilon = 1e-6; // Dimensionless: beta uses normalized stencil values.
        const std::array<double,3> optimal = {{0.1,0.6,0.3}};
        std::array<double,3> weights;
        double sum=0;
        for (int k=0; k<3; ++k) {
            const double denominator = epsilon+beta[k];
            weights[k] = optimal[k]/(denominator*denominator);
            sum += weights[k];
        }
        for (double& weight:weights) weight/=sum;
        return weights;
    }

    inline double combination(const std::array<double,3>& candidates,
                              const std::array<double,3>& weights)
    {
        return weights[0]*candidates[0]+weights[1]*candidates[1]+weights[2]*candidates[2];
    }

    struct FaceStates {
        double left, right;
    };

    inline FaceStates reconstruct(double im2, double im1, double i,
                                  double ip1, double ip2, double ip3,
                                  bool uniformStencil = true)
    {
        // The standard coefficients do not apply to half-width endpoint cells.
        // Also use this closure for extrapolated outflow and undersized grids.
        if (!uniformStencil) return {i,ip1};
        return {combination(leftCandidates(im2,im1,i,ip1,ip2),
                            nonlinearWeights(smoothness(im2,im1,i,ip1,ip2))),
                combination(rightCandidates(im1,i,ip1,ip2,ip3),
                            nonlinearWeights(smoothness(ip3,ip2,ip1,i,im1)))};
    }

    // Common normalized smoothness metrics across the conserved components.
    // This avoids giving rho and rhoY1+rhoY2+rhoY3 different reconstructions.
    // A stencil flagged by any component is downweighted for every component.
    template<std::size_t N>
    inline std::array<FaceStates,N> reconstructSystem(
        const std::array<std::array<double,6>,N>& values, bool uniformStencil = true,
        int vectorFirst = -1)
    {
        std::array<FaceStates,N> result;
        const bool hasVector = vectorFirst >= 0 && std::size_t(vectorFirst+1)<N;
        std::array<double,3> betaL={{0,0,0}}, betaR={{0,0,0}};
        if (uniformStencil) {
            for (std::size_t component=0; component<N; ++component) {
                if (hasVector && (int(component)==vectorFirst || int(component)==vectorFirst+1)) continue;
                const auto& q=values[component];
                const auto left=smoothness(q[0],q[1],q[2],q[3],q[4]);
                const auto right=smoothness(q[5],q[4],q[3],q[2],q[1]);
                for (int k=0; k<3; ++k) {
                    betaL[k]=std::max(betaL[k],left[k]);
                    betaR[k]=std::max(betaR[k],right[k]);
                }
            }
        }
        // Treat a vector pair as a vector: sum its quadratic indicators and
        // normalize by vector magnitude. Independent component normalization
        // would change the common weights when the velocity direction rotates.
        if (uniformStencil && hasVector) {
            const auto& x=values[vectorFirst];
            const auto& y=values[vectorFirst+1];
            double scaleL=0,scaleR=0;
            for(int i=0;i<5;++i) {
                scaleL=std::max(scaleL,std::hypot(x[i],y[i]));
                scaleR=std::max(scaleR,std::hypot(x[i+1],y[i+1]));
            }
            const auto lx=smoothnessWithScale(x[0],x[1],x[2],x[3],x[4],scaleL);
            const auto ly=smoothnessWithScale(y[0],y[1],y[2],y[3],y[4],scaleL);
            const auto rx=smoothnessWithScale(x[5],x[4],x[3],x[2],x[1],scaleR);
            const auto ry=smoothnessWithScale(y[5],y[4],y[3],y[2],y[1],scaleR);
            for(int k=0;k<3;++k) {
                betaL[k]=std::max(betaL[k],lx[k]+ly[k]);
                betaR[k]=std::max(betaR[k],rx[k]+ry[k]);
            }
        }
        const auto weightsL=nonlinearWeights(betaL), weightsR=nonlinearWeights(betaR);
        for (std::size_t k=0; k<N; ++k) {
            const auto& q=values[k];
            result[k] = uniformStencil
                ? FaceStates{combination(leftCandidates(q[0],q[1],q[2],q[3],q[4]),weightsL),
                             combination(rightCandidates(q[1],q[2],q[3],q[4],q[5]),weightsR)}
                : FaceStates{q[2],q[3]};
        }
        return result;
    }
}
