// Compile with -DTEST_THINC to compare with the baseline using identical data.
#ifdef TEST_THINC
#define MECHANICS2DMERGETHINC_NO_MAIN
#include "../src/mechanics2DMergeTHINC.C"
#else
#define MECHANICS2DMERGE_NO_MAIN
#include "../src/mechanics2DMerge.C"
#endif
int main(int argc,char** argv) {
    const Grid g{60,6,120,12};Settings settings;State state(8,g.size());
#ifdef TEST_THINC
    if(argc==2)settings.thincBeta=std::stod(argv[1]);
#endif
    for(int j=0;j<g.ny;++j)for(int i=0;i<g.nx;++i)
        state.col(g.index(i,j))=mixture(i<30?0:1,params::rho_init,300,500,0);
    double time=0;long fallbacks=0;
    while(time<.01-1e-15) {
        const auto rate=evaluate(state,g,settings);double dt=std::min(.01-time,.9*rate.stepLimit);
        bool accepted=false;
        for(int retry=0;retry<24 && !accepted;++retry) {
            for(bool high:{true,false})try {
                auto next=RK3Step(state,g,settings,dt,high,high?rate:evaluate(state,g,settings,false));
                state=std::move(next.state);accepted=true;if(!high)++fallbacks;break;
            }catch(const std::runtime_error&) {}
            if(!accepted)dt*=.5;
        }
        if(!accepted)throw std::runtime_error("Transport test failed to advance");
        time+=dt;
    }
    double width=0,Terror=0,Perror=0;
    for(int p=0;p<g.size();++p) {
        const double a=state(7,p);width+=4*a*(1-a)*g.dx()/g.ny;
        if(a>0){auto q=decode(state.col(p));Terror=std::max(Terror,std::abs(q.T-300));Perror=std::max(Perror,std::abs(q.p));}
    }
    if(Terror>1e-7 || Perror>1e-7)throw std::runtime_error("Uniform material thermodynamics changed");
    std::cout<<std::setprecision(17)<<width<<' '<<Terror<<' '<<Perror<<' '<<fallbacks<<'\n';
}
