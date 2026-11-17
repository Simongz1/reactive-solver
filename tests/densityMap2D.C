#define MECHANICS2DMERGETHINC_NO_MAIN
#include "../src/mechanics2DMergeTHINC.C"
#include <cassert>
int main(int argc,char** argv) {
    if(argc!=2)throw std::invalid_argument("Pass a temporary test directory");
    const std::string path=std::string(argv[1])+"/density_test.csv";
    {std::ofstream f(path);f<<"x,y,rho\n1,1,4\n0,0,0\n1,0,2\n0,1,3\n";}
    auto map=densityMap2D::read(path);
    assert(map.at(0,0)==0 && map.at(.9,.1)==2 && map.at(1,1)==4);
    bool failed=false;try{map.at(2,0);}catch(const std::invalid_argument&){failed=true;}assert(failed);
    {std::ofstream f(path);f<<"x,y,rho\n0,0,1\n0,0,2\n0,1,3\n1,1,4\n";}
    failed=false;try{densityMap2D::read(path);}catch(const std::invalid_argument&){failed=true;}assert(failed);
    // Independently verify all three shock jump conditions at non-reference density.
    for(double factor:{.98,1.,1.02}) {
        const double rho0=factor*params::rho_init,D=7000;
        const auto q=downstreamMappedState(D,rho0);
        auto near=[](double a,double b){assert(std::abs(a-b)<1e-8*std::max({1.,std::abs(a),std::abs(b)}));};
        const double p0=-pressure::value(rho0,300,false);
        near(rho0*D,q.density*(D-q.velocity));
        near(q.pressure-p0,rho0*D*q.velocity);
        near(q.pressure,-pressure::value(q.density,q.temperature,false));
        near(params::cv*q.temperature+referenceEnergy(q.density,false)
            -(params::cv*300+referenceEnergy(rho0,false)),
            .5*(q.pressure+p0)*(1/rho0-1/q.density));
    }
    std::cout<<"Density-map and mapped shock tests passed\n";
}
