#define MECHANICS2DMERGETHINC_NO_MAIN
#include "../src/mechanics2DMergeTHINC.C"
#include <cassert>
void near(double a,double b,double tol=1e-10) {
    if(!std::isfinite(a) || !std::isfinite(b) || std::abs(a-b)>tol*std::max({1.0,std::abs(a),std::abs(b)}))
        throw std::runtime_error("Comparison failed: "+std::to_string(a)+" vs "+std::to_string(b));
}
int main() {
    // Independently integrate the reconstructed tanh profile: its mean must
    // equal the supplied cell average for increasing and decreasing stencils.
    for(double beta:{1.6,2.2,3.0})for(double mean:{.01,.1,.3,.5,.8,.99})for(int sign:{-1,1}) {
        const auto edges=THINC::reconstruct(sign>0?0:1,mean,sign>0?1:0,beta);
        const double a=std::atanh(sign*(2*edges.left-1));
        const double integral=.5*(1+sign*(std::log(std::cosh(a+beta))-std::log(std::cosh(a)))/beta);
        near(integral,mean,1e-12);
        assert(edges.left>=0 && edges.left<=1 && edges.right>=0 && edges.right<=1);
        near(edges.right,.5*(1+sign*std::tanh(a+beta)),1e-12);
    }
    near(THINC::reconstruct(0,.8,.3).left,.8);
    near(THINC::reconstruct(1,1,1).right,1);

    Grid g{31,29,120,120};Settings settings; // Chemistry deliberately stays enabled.
    auto geom=levelSet2D::geometry(g,levelSet2D::circle(g,60,60,20));
    State s(8,g.size());
    for(int p=0;p<g.size();++p)s.col(p)=mixture(geom.fraction[p],params::rho_init,300,500,-200);
    auto rate=evaluate(s,g,settings);
    Step result;
    try {result=RK3Step(s,g,settings,1e-6,true,rate);}
    catch(const std::runtime_error&) {result=RK3Step(s,g,settings,1e-6,false,evaluate(s,g,settings,false));}
    double maxError=0;
    for(int p=0;p<g.size();++p)if(result.state(7,p)>0) {
        const auto q=decode(result.state.col(p));
        near(q.rho/params::rho_init,1,1e-10);near(q.T,300,1e-10);near(q.u,500);near(q.v,-200);
        maxError=std::max(maxError,std::abs(q.T-300));
    }
    std::cout<<"Uniform moving cavity: max temperature error "<<maxError<<" K\n";
    // Nonuniform material verifies finite-volume balance including chemistry and
    // thermal flux. Check every conserved component, not only total mass.
    for(int j=0;j<g.ny;++j)for(int i=0;i<g.nx;++i) {
        const double rho=params::rho_init*(1+.01*std::sin(i*.2));
        s.col(g.index(i,j))=mixture(1,rho,900+2*j,50*std::sin(i*.3),40*std::cos(j*.2));
    }
    rate=evaluate(s,g,settings);result=RK3Step(s,g,settings,std::min(1e-7,.1*rate.stepLimit),true,rate);
    const Eigen::Matrix<double,7,1> change=(result.state.topRows<7>()-s.topRows<7>()).rowwise().sum()*g.dx()*g.dy();
    for(int k:{3,4,5})near(change[k],result.boundary[k],1e-9);
    near(change[6],result.boundary[6]+result.reactionHeat,1e-9);
    for(int p=0;p<g.size();++p){near(result.state(7,p),1);near(result.state(0,p)+result.state(1,p)+result.state(2,p),result.state(3,p),1e-15);}
    // Reflection across x/y: same WENO stencils and grouped vector smoothness.
    State transposed(8,g.size());Grid gt{g.ny,g.nx,g.ly,g.lx};
    for(int j=0;j<g.ny;++j)for(int i=0;i<g.nx;++i){transposed.col(gt.index(j,i))=s.col(g.index(i,j));std::swap(transposed(4,gt.index(j,i)),transposed(5,gt.index(j,i)));}
    const auto rt=evaluate(transposed,gt,settings);
    for(int j=0;j<g.ny;++j)for(int i=0;i<g.nx;++i)for(int k=0;k<8;++k)
        near(rate.rhs(k,g.index(i,j)),rt.rhs(k==4?5:k==5?4:k,gt.index(j,i)),1e-8);
    // Mixed-cell closure must not change any extensive conserved quantity.
    State compacted(8,g.size());
    for(int p=0;p<g.size();++p)compacted.col(p)=mixture(.8,1.5*params::rho_init,400,200,-100);
    const State before=compacted;equilibrate(compacted,g);
    near((compacted.topRows<7>()-before.topRows<7>()).norm(),0,1e-15);
    for(int p=0;p<g.size();++p){near(compacted(7,p),1);decode(compacted.col(p));}
    for(int p=0;p<g.size();++p)compacted.col(p)=mixture(.8,.8*params::rho_init,300,200,-100);
    const State expanded=compacted;equilibrate(compacted,g);
    near((compacted.topRows<7>()-expanded.topRows<7>()).norm(),0,1e-15);
    for(int p=0;p<g.size();++p){assert(compacted(7,p)>0 && compacted(7,p)<1);near(decode(compacted.col(p)).p,0,1e-9);}
    // Active, appreciable chemistry: verify energy accounting at high T.
    for(int p=0;p<g.size();++p)s.col(p)=mixture(1,params::rho_init,2200,0,0);
    rate=evaluate(s,g,settings);result=RK3Step(s,g,settings,.1*rate.stepLimit,true,rate);
    assert(result.state(0,0)<s(0,0) && result.reactionHeat!=0);
    near((result.state.row(6)-s.row(6)).sum()*g.dx()*g.dy(),result.reactionHeat,1e-9);
    // Thermal timestep is intrinsic: it must not collapse for tiny alpha.
    const double fullLimit=rate.stepLimit;
    State tiny=s*1e-80;
    const auto tinyRate=evaluate(tiny,g,settings);
    near(tinyRate.stepLimit/fullLimit,1,1e-10);
    // Smooth full-material species transport retains the original WENO5 stencil.
    double previous=0;
    for(int n:{41,81,161}) {
        Grid waveGrid{n,6,1,1};State wave(8,waveGrid.size());
        const double pi=std::acos(-1.0),h=waveGrid.dx(),rho=params::rho_init;
        for(int j=0;j<waveGrid.ny;++j)for(int i=0;i<n;++i) {
            auto cell=mixture(1,rho,300,2,0);
            const double y=.5+.1*std::sin(2*pi*(i+.5)*h)*std::sin(pi*h)/(pi*h);
            cell[0]=rho*y;cell[2]=rho*(1-y);wave.col(waveGrid.index(i,j))=cell;
        }
        const auto rw=evaluate(wave,waveGrid,settings);
        const double r1=reactionRates::computeArrhenius(300,params::R,params::E1,params::Z1);
        double error=0;
        for(int i=4;i<n-4;++i) {
            const int p=waveGrid.index(i,2);
            const double exact=-2*.1*(std::sin(2*pi*(i+1)*h)-std::sin(2*pi*i*h))/h;
            error=std::max(error,std::abs((rw.rhs(0,p)+r1*wave(0,p))/rho-exact));
        }
        std::cout<<"WENO n="<<n<<" error="<<error<<"\n";
        if(previous && previous/error<25)throw std::runtime_error("Lost WENO5 convergence");
        previous=error;
    }
    std::cout<<"Passed: translating vacuum, conservative mass/momentum/energy with active chemistry, species balance, full-material fraction invariant, axis exchange, conservative closure, hot chemistry, WENO5 convergence.\n";
}
