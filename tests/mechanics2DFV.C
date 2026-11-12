// Build with the same include/library paths as mechanics2DFV.C.
#define MECHANICS2DFV_NO_MAIN
#include "../src/mechanics2DFV.C"
#include <cassert>

void near(double a, double b, double tolerance = 2e-11)
{
    if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a-b) > tolerance*std::max({1.0,std::abs(a),std::abs(b)}))
        throw std::runtime_error("Numerical comparison failed: " + std::to_string(a) + " vs " + std::to_string(b));
}

// A nonuniform thermodynamic state flowing outward through all four sides.
State sampleState()
{
    State s(nvars,Eigen::MatrixXd::Zero(nx,ny));
    for (int j=0; j<ny; ++j) for (int i=0; i<nx; ++i) {
        const double rho = params::rho_init*(1+0.01*(i+j));
        const double y1=0.6+0.01*i, y2=0.1+0.01*j;
        const double u=0.4*(i-(nx-1)/2.0);
        const double v=0.3*(j-(ny-1)/2.0);
        s[0](i,j)=rho*y1; s[1](i,j)=rho*y2; s[2](i,j)=rho*(1-y1-y2);
        s[3](i,j)=rho; s[4](i,j)=rho*u; s[5](i,j)=rho*v;
        s[6](i,j)=rho*(params::cv*(800+11*i+17*j)
            + y1*referenceEnergy(rho,false)+(1-y1)*referenceEnergy(rho,true)+0.5*(u*u+v*v));
    }
    return s;
}

void smoothTransport()
{
    // A material-composition wave at constant rho, T=300 and velocity.
    // At rho_init and T=300 both phases have zero gauge pressure/reference energy.
    // Test the actual reconstructed flux divergence, not just the header formulas.
    double previous=0;
    for(int nodes:{41,81,161}) {
        nx=nodes; ny=9; dx=1.0/(nx-1); dy=.1;
        const double rho=params::rho_init, velocity=2, pi=std::acos(-1.0);
        State s(nvars,Eigen::MatrixXd::Zero(nx,ny));
        for(int j=0;j<ny;++j) for(int i=0;i<nx;++i) {
            const double y1=.5+.1*std::sin(2*pi*i*dx)*std::sin(pi*dx)/(pi*dx);
            s[0](i,j)=rho*y1; s[2](i,j)=rho*(1-y1); s[3](i,j)=rho;
            s[4](i,j)=rho*velocity;
            s[6](i,j)=rho*(params::cv*300+.5*velocity*velocity);
        }
        const auto r=assembleResidual(s);
        const double rate=reactionRates::computeArrhenius(300,params::R,params::E1,params::Z1);
        double error=0;
        for(int i=4;i<nx-4;++i) {
            const double exact=-velocity*.1*(std::sin(2*pi*(i+.5)*dx)-std::sin(2*pi*(i-.5)*dx))/dx;
            error=std::max(error,std::abs((r[0](i,4)+rate*s[0](i,4))/rho-exact));
            for(int j=0;j<ny;++j) for(int k=0;k<nvars;++k) near(r[k](i,j),r[k](i,4));
        }
        if(previous && previous/error<25)
            throw std::runtime_error("Smooth transport did not approach fifth order");
        previous=error;
    }
}

void conservationAndSymmetry()
{
    nx=13; ny=11; dx=2; dy=3;
    auto s=sampleState();
    const auto r=assembleResidual(s);
    double mass=0, energy=0, energyScale=0;
    for (int j=0; j<ny; ++j) for (int i=0; i<nx; ++i) {
        const double area=dx*dy*(i==0 || i==nx-1 ? 0.5 : 1)*(j==0 || j==ny-1 ? 0.5 : 1);
        near(r[0](i,j)+r[1](i,j)+r[2](i,j),r[3](i,j));
        mass+=area*r[3](i,j);
        const double T=temperature(s,i,j), rho=s[3](i,j);
        const double r1=reactionRates::computeArrhenius(T,params::R,params::E1,params::Z1);
        const double r2=reactionRates::computeArrhenius(T,params::R,params::E2,params::Z2);
        const double source=params::rho_init/rho*(reactionRates::computeReactionHeat(params::a1,params::b1,T)*r1*s[0](i,j)
            + reactionRates::computeReactionHeat(params::a2,params::b2,T)*r2*s[1](i,j));
        energy+=area*(r[6](i,j)-source);
        energyScale+=area*std::abs(r[6](i,j)-source);
    }
    // Integrate physical outward boundary fluxes independently of the residual.
    double massOut=0, energyOut=0;
    auto accumulate = [&](int i, int j, int momentum, double signedLength) {
        const double velocity=s[momentum](i,j)/s[3](i,j);
        massOut+=signedLength*s[momentum](i,j);
        energyOut+=signedLength*velocity*(s[6](i,j)-stress(s,i,j));
    };
    for(int j=0;j<ny;++j) {
        const double length=dy*(j==0 || j==ny-1 ? .5 : 1);
        accumulate(0,j,4,-length); accumulate(nx-1,j,4,length);
    }
    for(int i=0;i<nx;++i) {
        const double length=dx*(i==0 || i==nx-1 ? .5 : 1);
        accumulate(i,0,5,-length); accumulate(i,ny-1,5,length);
    }
    assert(massOut>0 && mass<0);
    near(mass,-massOut);
    near((energy+energyOut)/std::max(1.0,energyScale),0);
    assert(r[4].row(0).norm()>0 && r[4].row(nx-1).norm()>0);
    assert(r[5].col(0).norm()>0 && r[5].col(ny-1).norm()>0);
    State transposed;
    for (const auto& field:s) transposed.push_back(field.transpose());
    std::swap(transposed[4],transposed[5]);
    std::swap(nx,ny); std::swap(dx,dy);
    auto rt=assembleResidual(transposed);
    std::swap(rt[4],rt[5]);
    for(int k=0;k<nvars;++k) for(int j=0;j<ny;++j) for(int i=0;i<nx;++i)
        near(rt[k](i,j),r[k](j,i));
}

void uniformMovingState()
{
    nx=13; ny=11; dx=2; dy=3; params::dt=1e-6;
    State s(nvars,Eigen::MatrixXd::Zero(nx,ny));
    const double rho=params::rho_init, u=2.0, v=-3.0;
    s[2].setConstant(rho); s[3].setConstant(rho); // Fully reacted: no sources.
    s[4].setConstant(rho*u); s[5].setConstant(rho*v);
    s[6].setConstant(rho*(params::cv*800+0.5*(u*u+v*v)));
    const auto residual=assembleResidual(s), original=s;
    for(const auto& field:residual) near(field.norm(),0);
    RK3Step(s);
    for(int k=0;k<nvars;++k) near((s[k]-original[k]).norm(),0);
}

void exodusRoundTrip(const char* path)
{
    nx=13; ny=11; dx=2; dy=3;
    const auto fields=outputFields(sampleState());
    ExodusOutput output(path,nx,ny,dx,dy,{"Y1","Y2","Y3","T","rho","vx","vy","rhoE","p"});
    output.write(0.125,fields); output.close();
    int cpu=8, disk=0; float version;
    int id=ex_open(path,EX_READ,&cpu,&disk,&version);
    assert(id>=0);
    double time; assert(ex_get_time(id,1,&time)==0); near(time,0.125);
    for(int k=0;k<9;++k) {
        Eigen::MatrixXd actual(nx,ny);
        assert(ex_get_var(id,1,EX_NODAL,k+1,1,nx*ny,actual.data())==0);
        for(int j=0;j<ny;++j) for(int i=0;i<nx;++i) near(actual(i,j),fields[k](i,j));
    }
    assert(ex_close(id)==0);
}

int main(int argc, char** argv)
{
    for (bool viscosity:{false,true}) {
        params::use_artificial_viscosity=viscosity;
        smoothTransport(); conservationAndSymmetry(); uniformMovingState();
    }
    assert(argc==2);
    exodusRoundTrip(argv[1]);
    std::cout << "Passed: smooth transport convergence, boundary mass/energy balance, species balance, evolving boundary momenta, uniform moving state, axis symmetry, Exodus readback.\n";
}
