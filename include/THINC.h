#pragma once
#include <algorithm>
#include <cmath>

// Cell-average-preserving THINC profile on [0,1]. Only monotone stencils
// are reconstructed; extrema and nearly pure cells keep their cell averages.
namespace THINC {
struct Traces { double left,right; };
inline Traces reconstruct(double previous,double average,double next,double beta=1.6) {
    const double lower=std::min(previous,next),range=std::abs(next-previous);
    if(!(beta>0) || !std::isfinite(beta) || beta>10)
        return {average,average};
    if(average<=1e-12 || average>=1-1e-12 || range<=1e-12
       || average<=lower || average>=lower+range)
        return {average,average};
    const double direction=next>previous?1:-1;
    const double C=(average-lower)/range;
    const double t=std::tanh(beta);
    const double A=(std::exp(direction*beta*(2*C-1))/std::cosh(beta)-1)/t;
    const double left=lower+.5*range*(1+direction*A);
    const double right=lower+.5*range*(1+direction*(t+A)/(1+A*t));
    // Guard roundoff in the analytic evaluation, not the evolved solution.
    return {std::clamp(left,lower,lower+range),std::clamp(right,lower,lower+range)};
}
}
