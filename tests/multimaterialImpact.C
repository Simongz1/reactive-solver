#define MECHANICS2DMULTIMATERIAL_NO_MAIN
#include "../src/mechanics2DMultimaterial.C"
#include <cassert>
int main() {
    auto rdx = pureState(0,params::rho_init,300,0,-750);
    auto q=decode(rdx);
    assert(std::abs(q.T-300)<1e-10 && q.v==-750);
    Grid grid{8,8,200,200};
    State state(stateFieldCount,grid.size());
    for(int c=0;c<grid.size();++c) state.col(c)=pureState(0,params::rho_init,300,100,-750);
    Settings s; s.openBoundaries=true; s.reactions=false; s.conduction=false;
    auto r=evaluate(state,grid,s);
    assert(r.rhs.cwiseAbs().maxCoeff()<1e-8);
    s.fixedCells={0};
    state.col(0)=pureState(1,multimaterialEOS::rhoPS);
    r=evaluate(state,grid,s);
    auto step=RK3Step(state,grid,s,std::min(1e-8,.01*r.stepLimit),false,r);
    assert(step.state(MomentumX,0)==0 && step.state(MomentumY,0)==0);
    Conserved delta=(step.state.topRows(conservedFieldCount)-state.topRows(conservedFieldCount)).rowwise().sum()*grid.dx()*grid.dy();
    assert((delta-step.boundary).cwiseAbs().maxCoeff()<1e-8);
}
