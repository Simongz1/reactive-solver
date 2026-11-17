// THINC interface reconstruction variant of mechanics2DMerge.C.
// All physics, fluxes, source terms, RK stages and boundaries match the baseline.
#include <cmath>
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "modelParams.h"
#include "pressure.h"
#include "computeReactionRates.h"
#include "WENO.h"
#include "THINC.h"
#include "densityMap2D.h"
#include "riemannMaterial2D.h"
#include "levelSet2D.h"
#include "ExodusOutput.h"

using Grid = levelSet2D::Grid;
using State = Eigen::Matrix<double, 8, Eigen::Dynamic>;
using Cell = Eigen::Matrix<double, 8, 1>;

struct Primitive { double rho,T,u,v,p,c; };
struct Settings {
    bool reactions = true;
    bool conduction = true;
    Cell inlet = Cell::Zero();
    std::vector<Cell> inletRows; // Optional mapped upstream state for each inlet row.
    bool driven = false;
    double thincBeta = 1.6; //default value
};

double referenceEnergy(double rho, bool reacted)
{
    const double compression = 1.0 - params::rho_init/rho;
    const double omega = reacted ? params::omega_reacted : params::omega_unreacted;
    const auto coldWork = [compression](double A, double R) {
        return A*std::exp(-R)*(std::expm1(R*compression)/R - compression);
    };
    const double A = reacted ? params::A_reacted : params::A_unreacted;
    const double B = reacted ? params::B_reacted : params::B_unreacted;
    const double R1 = reacted ? params::R1_reacted : params::R1_unreacted;
    const double R2 = reacted ? params::R2_reacted : params::R2_unreacted;
    return (coldWork(A,R1) + coldWork(B,R2))/params::rho_init
         + omega*params::cv*300.0*std::log(params::rho_init/rho);
}

// Unreacted Hugoniot from 300 K and zero gauge pressure.
// eta = 1 - rho0/rho. The Rayleigh line gives p = rho0*D^2*eta.
struct ShockState {
    double density, velocity, temperature, pressure;
};

ShockState downstreamState(double speed)
{
    const double rho0 = params::rho_init, cv = params::cv;
    const double omega = params::omega_unreacted;
    const double r1 = params::R1_unreacted, r2 = params::R2_unreacted;
    const double a = params::A_unreacted * std::exp(-r1);
    const double b = params::B_unreacted * std::exp(-r2);
    const double slope = rho0 * speed * speed;
    const double acousticSlope = a * r1 + b * r2 + rho0 * cv * omega * omega * 300;

    if (!std::isfinite(slope) || slope <= acousticSlope)
        throw std::invalid_argument("Shock speed must exceed the upstream adiabatic sound speed");

    //compute the post-shock temperature rise from the reference state
    auto temperatureRise = [&](double eta) {
        const double coldWork = a*(std::expm1(r1*eta)/r1 - eta)
                              + b*(std::expm1(r2*eta)/r2 - eta);
        return (0.5*slope*eta*eta - coldWork)/(rho0*cv)
             - omega*300*std::log1p(-eta);
    };

    auto pressureMismatch = [&](double eta) {
        const double cold = a*std::expm1(r1*eta) + b*std::expm1(r2*eta);
        return cold + omega*rho0*cv*temperatureRise(eta)/(1-eta) - slope*eta;
    };

    double low = 0, high = 2/(2+omega);
    if (!(pressureMismatch(high) > 0))
        throw std::invalid_argument("No finite compressive shock state could be bracketed");
    for (int n = 0; n < 80; ++n) {
        const double mid = (low+high)/2;
        if (pressureMismatch(mid) < 0) low = mid;
        else high = mid;
    }
    const double eta = (low+high)/2;
    const double rho = rho0/(1-eta);
    const double temperature = 300 + temperatureRise(eta);
    const double pressureBehind = -pressure::value(rho,temperature,false);
    if (!(temperature >= 300 && std::isfinite(temperature) && std::isfinite(pressureBehind)))
        throw std::invalid_argument("Shock solution has invalid temperature or pressure");
    return {rho,speed*eta,temperature,pressureBehind};
}

// Rankine-Hugoniot solve for arbitrary mapped intrinsic upstream density.
// The EOS reference density remains a material constant.
ShockState downstreamMappedState(double speed,double rho0)
{
    if(rho0==params::rho_init)return downstreamState(speed);
    const double T0=300, cv=params::cv;
    const double p0=-pressure::value(rho0,T0,false);
    const double e0=cv*T0+referenceEnergy(rho0,false);
    const double pT=-pressure::dT(rho0,T0,false);
    const double c2=pressure::drho(rho0,T0,false)+T0*pT*pT/(rho0*rho0*cv);
    if(!std::isfinite(c2) || c2<0 || speed*speed<=c2)
        throw std::invalid_argument("Shock speed must exceed local upstream sound speed in density map");
    auto stateAt=[&](double eta) {
        const double rho=rho0/(1-eta),p=p0+rho0*speed*speed*eta;
        const double T=(e0+.5*(p+p0)*eta/rho0-referenceEnergy(rho,false))/cv;
        return ShockState{rho,speed*eta,T,p};
    };
    auto mismatch=[&](double eta) {
        const auto q=stateAt(eta);
        return -pressure::value(q.density,q.temperature,false)-q.pressure;
    };
    double low=1e-8,high=low;
    if(!(mismatch(low)<0))throw std::invalid_argument("Cannot bracket mapped compressive shock");
    bool bracket=false;
    for(int n=1;n<=10000;++n){high=.999*n/10000;
        const double f=mismatch(high);
        if(!std::isfinite(f))break;
        if(f>0){bracket=true;break;}low=high;
    }
    if(!bracket)throw std::invalid_argument("No admissible mapped shock root found");
    for(int n=0;n<80;++n){double mid=.5*(low+high);if(mismatch(mid)>0)high=mid;else low=mid;}
    const auto q=stateAt(.5*(low+high));
    if(!std::isfinite(q.temperature) || q.temperature<=0)
        throw std::invalid_argument("Mapped shock temperature is invalid");
    return q;
}

Primitive primitive(const Eigen::VectorXd& U)
{
    const double rho=U[3];
    if(!U.allFinite() || !(rho>0)) throw std::runtime_error("Invalid material density");
    const double u=U[4]/rho,v=U[5]/rho,Y1=U[0]/rho;
    for(int k=0;k<3;++k)
        if(U[k]<-1e-10*rho || U[k]>(1+1e-10)*rho) throw std::runtime_error("Invalid material species");
    const double T=(U[6]/rho-.5*(u*u+v*v)-Y1*referenceEnergy(rho,false)
        -(1-Y1)*referenceEnergy(rho,true))/params::cv;
    const double p=-Y1*pressure::value(rho,T,false)-(1-Y1)*pressure::value(rho,T,true);
    const double dpdrho=Y1*pressure::drho(rho,T,false)+(1-Y1)*pressure::drho(rho,T,true);
    const double dpdT=-Y1*pressure::dT(rho,T,false)-(1-Y1)*pressure::dT(rho,T,true);
    const double c2=dpdrho+T*dpdT*dpdT/(rho*rho*params::cv);
    if(!std::isfinite(T) || T<=0 || !std::isfinite(p) || !std::isfinite(c2) || c2<0) {
        std::ostringstream message;
        message << std::setprecision(17)
            << "Invalid material thermodynamic state: T=" << T << " K, c2=" << c2
            << ", p=" << p << " GPa, rho=" << rho
            << ", specific total energy=" << U[6]/rho
            << ", specific kinetic energy=" << .5*(u*u+v*v)
            << ", reference energy=" << Y1*referenceEnergy(rho,false)+(1-Y1)*referenceEnergy(rho,true);
        throw std::runtime_error(message.str());
    }
    return {rho,T,u,v,p,std::sqrt(c2)};
}
Eigen::VectorXd materialState(double rho,double T,double u,double v)
{
    Eigen::VectorXd U=Eigen::VectorXd::Zero(7);
    U[0]=rho;U[3]=rho;U[4]=rho*u;U[5]=rho*v;
    U[6]=rho*(params::cv*T+referenceEnergy(rho,false)+.5*(u*u+v*v));
    return U;
}
Eigen::VectorXd flux(const Eigen::VectorXd& U,const Primitive& q,bool alongX)
{
    const double velocity=alongX ? q.u : q.v;
    Eigen::VectorXd F=velocity*U;
    F[alongX ? 4 : 5]+=q.p;F[6]+=velocity*q.p;
    return F;
}


Cell mixture(double alpha,double rho,double T,double u,double v) {
    Cell q; q.head<7>()=alpha*materialState(rho,T,u,v);q[7]=alpha;return q;
}
Primitive decode(const Cell& q) {
    if(!q.allFinite() || q[7]<0 || q[7]>1+2e-12)
        throw std::runtime_error("Invalid material fraction");
    if(q[7]==0) {
        if(q.head<7>().squaredNorm()!=0) throw std::runtime_error("Mass/energy in empty cell");
        return {0,300,0,0,0,0};
    }
    return primitive(q.head<7>()/q[7]);
}
void validate(const State& s,const Grid& g) {
    for(int p=0;p<g.size();++p) {
        try {decode(s.col(p));}
        catch(const std::runtime_error& e) {
            std::ostringstream text;text<<std::setprecision(17)<<e.what()<<" at ("<<p%g.nx<<','<<p/g.nx
                <<"), state="<<s.col(p).transpose();throw std::runtime_error(text.str());
        }
    }
}
// Instantaneous material/vacuum pressure equilibrium in mixed cells.
// Vacuum exerts p=0, so changing occupied volume performs no external work.
// Solve only alpha; the seven extensive conserved quantities are untouched.
// Once available volume is exhausted, alpha=1 and ordinary compressible flow
// supports positive pressure. This is a local scalar EOS root, not contact.
void equilibrate(State& s,const Grid& g) {
    for(int p=0;p<g.size();++p) {
        const double alpha=s(7,p),mass=s(3,p);
        if(alpha==0 || alpha>=1)continue;
        if(mass<=0)throw std::runtime_error("Invalid mass in pressure relaxation");
        const double y=s(0,p)/mass,u=s(4,p)/mass,v=s(5,p)/mass;
        const double internal=s(6,p)/mass-.5*(u*u+v*v);
        auto thermodynamics=[&](double a) {
            const double rho=mass/a;
            const double T=(internal-y*referenceEnergy(rho,false)-(1-y)*referenceEnergy(rho,true))/params::cv;
            const double pressureValue=-y*pressure::value(rho,T,false)-(1-y)*pressure::value(rho,T,true);
            return std::pair<double,double>{pressureValue,T};
        };
        // Avoid division by vanishing alpha: normalize the search variable by
        // its current value. Pressure equilibrium is independent of cell mass.
        const auto atFull=thermodynamics(1);
        if(atFull.first>=0 && atFull.second>0) {s(7,p)=1;continue;}
        const auto current=thermodynamics(alpha);
        if(std::abs(current.first)<1e-12 && current.second>0)continue;
        double lower=alpha;
        auto lowState=current;
        for(int iteration=0;lowState.first<0 && iteration<80;++iteration) {
            lower*=.5;lowState=thermodynamics(lower);
        }
        if(!std::isfinite(lowState.first) || lowState.first<0 || atFull.first>0)
            throw std::runtime_error("Cannot bracket material/vacuum pressure equilibrium");
        // Bisection in logarithmic volume prevents loss of relative precision
        // when conservative transport creates a very small material fraction.
        double lo=std::log(lower),hi=0;
        for(int iteration=0;iteration<70;++iteration) {
            const double mid=.5*(lo+hi),a=std::exp(mid);
            if(thermodynamics(a).first>0)lo=mid;else hi=mid;
        }
        s(7,p)=std::exp(.5*(lo+hi));
        decode(s.col(p));
    }
}
struct Evaluation {
    State rhs;
    double stepLimit=std::numeric_limits<double>::infinity();
    // Integrated net influx of conserved fields, and reaction energy rate.
    Eigen::Matrix<double,7,1> boundary=Eigen::Matrix<double,7,1>::Zero();
    double reactionHeat=0;
    long faceFallbacks=0;
};

Evaluation evaluate(const State& s,const Grid& g,const Settings& settings,bool highOrder=true) {
    Evaluation out;out.rhs=State::Zero(8,g.size());
    std::vector<Primitive> q(g.size());
    std::vector<int> owner(g.size(),-1);std::queue<int> pending;
    double inverseDt=0;
    for(int p=0;p<g.size();++p) {
        q[p]=decode(s.col(p));
        if(s(7,p)>0) {
            owner[p]=p;pending.push(p);
            inverseDt=std::max(inverseDt,(std::abs(q[p].u)+q[p].c)/g.dx()
                +(std::abs(q[p].v)+q[p].c)/g.dy()
                +2*(settings.conduction?params::k:0)/(q[p].rho*params::cv)*(1/(g.dx()*g.dx())+1/(g.dy()*g.dy())));
        }
    }
    if(pending.empty()) throw std::runtime_error("No material remains");
    // Empty cells carry no thermodynamics. Extend velocity only for alpha's
    // nonconservative transport; never assign vacuum material mass or energy.
    while(!pending.empty()) {
        const int p=pending.front();pending.pop();const int i=p%g.nx,j=p/g.nx;
        for(auto d:{std::pair<int,int>{-1,0},{1,0},{0,-1},{0,1}}) {
            const int a=i+d.first,b=j+d.second;
            if(a<0 || a>=g.nx || b<0 || b>=g.ny) continue;
            const int r=g.index(a,b);if(owner[r]>=0) continue;
            owner[r]=owner[p];pending.push(r);
        }
    }
    auto inletCell=[&](int j)->const Cell& {
        return settings.inletRows.empty()?settings.inlet:settings.inletRows.at(std::clamp(j,0,g.ny-1));
    };
    auto sample=[&](int i,int j)->Cell {
        if(i<0 && settings.driven) return inletCell(j);
        return s.col(g.index(std::clamp(i,0,g.nx-1),std::clamp(j,0,g.ny-1)));
    };
    auto face=[&](int i,int j,bool x) {
        const Primitive inlet=settings.driven?decode(inletCell(j)):Primitive{};
        std::array<std::array<double,6>,8> stencil;
        for(int n=0;n<6;++n) {
            const Cell value=sample(i+(x?n-3:0),j+(x?0:n-3));
            for(int k=0;k<8;++k) stencil[k][n]=value[k];
        }
        const bool interior=x ? i>=3 && i<=g.nx-3 : j>=3 && j<=g.ny-3;
        // Start interface stencils with the existing first-order safety closure;
        // monotone mixed cells receive THINC traces below.
        // In full material, preserve exactly the WENO5 system stencil.
        bool fullMaterial=true;
        for(double a:stencil[7])fullMaterial=fullMaterial && a>=1-1e-12;
        if(highOrder && interior && !fullMaterial)++out.faceFallbacks;
        const auto traces=WENO::reconstructSystem<8>(stencil,highOrder && interior && fullMaterial,4);
        Cell left,right;for(int k=0;k<8;++k){left[k]=traces[k].left;right[k]=traces[k].right;}
        // THINC replaces only the baseline piecewise-constant interface traces.
        // Use the SAME alpha profile for all extensive fields, keeping each
        // cell's intrinsic rho, velocity, composition and specific energy fixed.
        // Thus reconstructed alpha does not spuriously alter T or pressure.
        if(highOrder && interior && !fullMaterial) {
            const auto alphaLeft=THINC::reconstruct(stencil[7][1],stencil[7][2],stencil[7][3],settings.thincBeta);
            const auto alphaRight=THINC::reconstruct(stencil[7][2],stencil[7][3],stencil[7][4],settings.thincBeta);
            for(int side=0;side<2;++side) {
                const int center=side+2;
                const double cellAlpha=stencil[7][center];
                const double faceAlpha=side==0?alphaLeft.right:alphaRight.left;
                if(cellAlpha<=1e-12 || cellAlpha>=1-1e-12)continue;
                Cell& trace=side==0?left:right;
                for(int k=0;k<7;++k)trace[k]=(stencil[k][center]/cellAlpha)*faceAlpha;
                trace[7]=faceAlpha;
            }
        }
        bool constantAlpha=true;
        for(double a:stencil[7])constantAlpha=constantAlpha && a==stencil[7][0];
        if(constantAlpha){left[7]=stencil[7][0];right[7]=stencil[7][0];}
        // Bound each trace by contracting the entire state toward its cell
        // average, preserving species/mass identities and alpha-rho consistency.
        for(int side=0;side<2;++side) {
            Cell& trace=side==0?left:right;Cell center;
            for(int k=0;k<8;++k)center[k]=stencil[k][side+2];
            double theta=1;
            if(trace[7]<0)theta=std::min(theta,center[7]/(center[7]-trace[7]));
            if(trace[7]>1)theta=std::min(theta,(1-center[7])/(trace[7]-center[7]));
            if(theta<1)trace=center+std::max(0.0,theta)*(trace-center);
        }
        Primitive a,b;
        try {a=decode(left);b=decode(right);}
        catch(const std::runtime_error&) {
            for(int k=0;k<8;++k){left[k]=stencil[k][2];right[k]=stencil[k][3];}
            a=decode(left);b=decode(right);++out.faceFallbacks;
        }
        const int l=g.index(std::clamp(i-(x?1:0),0,g.nx-1),std::clamp(j-(x?0:1),0,g.ny-1));
        const int r=g.index(std::clamp(i,0,g.nx-1),std::clamp(j,0,g.ny-1));
        const Primitive& extL=(x && i==0 && settings.driven)?inlet:q[owner[l]];
        const Primitive& extR=q[owner[r]];
        if(left[7]==0) {a.u=extL.u;a.v=extL.v;}
        if(right[7]==0) {b.u=extR.u;b.v=extR.v;}
        const double ua=x?a.u:a.v,ub=x?b.u:b.v;
        const double speed=std::max(std::abs(ua)+a.c,std::abs(ub)+b.c);
        inverseDt=std::max(inverseDt,2*speed/(x?g.dx():g.dy()));
        Cell fl=ua*left,fr=ub*right;
        fl[x?4:5]+=left[7]*a.p;fr[x?4:5]+=right[7]*b.p;
        fl[6]+=ua*left[7]*a.p;fr[6]+=ub*right[7]*b.p;
        Cell F=.5*(fl+fr)-.5*speed*(right-left);
        // Centered heat stencil, zero heat flux into exact vacuum. Use cell
        // temperatures rather than two traces at the same reconstructed point.
        const double al=(x && i==0 && settings.driven)?inletCell(j)[7]:s(7,l),ar=s(7,r);
        const double conductivity=(settings.conduction && al>0 && ar>0)?2*al*ar/(al+ar)*params::k:0;
        F[6]-=conductivity*(q[r].T-((x && i==0 && settings.driven)?inlet.T:q[l].T))/(x?g.dx():g.dy());
        const double spacing=x?g.dx():g.dy();
        const bool hasLeft=x?i>0:j>0,hasRight=x?i<g.nx:j<g.ny;
        if(hasLeft) out.rhs.col(l)-=F/spacing;
        if(hasRight) out.rhs.col(r)+=F/spacing;
        if(!hasLeft) out.boundary+=F.head<7>()*(x?g.dy():g.dx());
        if(!hasRight) out.boundary-=F.head<7>()*(x?g.dy():g.dx());
        // alpha_t + div(alpha*u) = alpha*div(u). Same interface velocity
        // on each side makes alpha=1 invariant and preserves uniform translation.
        const double velocity=.5*(ua+ub);
        if(hasLeft) out.rhs(7,l)+=s(7,l)*velocity/spacing;
        if(hasRight) out.rhs(7,r)-=s(7,r)*velocity/spacing;
    };
    for(int j=0;j<g.ny;++j) for(int i=0;i<=g.nx;++i) face(i,j,true);
    for(int j=0;j<=g.ny;++j) for(int i=0;i<g.nx;++i) face(i,j,false);
    out.stepLimit=.08/std::max(inverseDt,1e-30);
    if(settings.reactions) for(int p=0;p<g.size();++p) if(s(7,p)>0) {
        const double T=q[p].T,rho=q[p].rho;
        const double r1=reactionRates::computeArrhenius(T,params::R,params::E1,params::Z1);
        const double r2=reactionRates::computeArrhenius(T,params::R,params::E2,params::Z2);
        const double heat=(params::rho_init/rho)*(
            reactionRates::computeReactionHeat(params::a1,params::b1,T)*r1*s(0,p)
           +reactionRates::computeReactionHeat(params::a2,params::b2,T)*r2*s(1,p));
        out.rhs(0,p)-=r1*s(0,p);out.rhs(1,p)+=r1*s(0,p)-r2*s(1,p);out.rhs(2,p)+=r2*s(1,p);
        out.rhs(6,p)+=heat;out.reactionHeat+=heat*g.dx()*g.dy();
        out.stepLimit=std::min(out.stepLimit,.08/std::max({r1,r2,1e-30}));
        // Control source-induced thermal changes as well as species depletion.
        const double specificHeat=(params::rho_init/rho)*(
            reactionRates::computeReactionHeat(params::a1,params::b1,T)*r1*(s(0,p)/s(3,p))
           +reactionRates::computeReactionHeat(params::a2,params::b2,T)*r2*(s(1,p)/s(3,p)));
        const double referenceChange=std::abs(r1*(s(0,p)/s(3,p))*(referenceEnergy(rho,false)-referenceEnergy(rho,true)));
        out.stepLimit=std::min(out.stepLimit,.05*params::cv*T/std::max(std::abs(specificHeat)+referenceChange,1e-30));
    }
    return out;
}
struct Step {
    State state;
    Eigen::Matrix<double,7,1> boundary;
    double reactionHeat;
    long faceFallbacks;
};
Step RK3Step(const State& s,const Grid& g,const Settings& settings,double dt,bool highOrder,const Evaluation& r0) {
    auto advance=[&](const State& a,const Evaluation& r)->State {
        if(dt>r.stepLimit*(1+1e-12)) {
            std::ostringstream message;message<<std::setprecision(17)<<"Stage CFL/source bound exceeded: dt="<<dt<<", bound="<<r.stepLimit;
            throw std::runtime_error(message.str());
        }
        State b=a+dt*r.rhs;equilibrate(b,g);validate(b,g);return b;
    };
    State a=advance(s,r0);const auto r1=evaluate(a,g,settings,highOrder);
    State b=.75*s+.25*advance(a,r1);equilibrate(b,g);validate(b,g);
    const auto r2=evaluate(b,g,settings,highOrder);
    State c=(1.0/3)*s+(2.0/3)*advance(b,r2);equilibrate(c,g);validate(c,g);
    return {std::move(c),dt*(r0.boundary/6+r1.boundary/6+2*r2.boundary/3),
        dt*(r0.reactionHeat/6+r1.reactionHeat/6+2*r2.reactionHeat/3),
        r0.faceFallbacks+r1.faceFallbacks+r2.faceFallbacks};
}

#ifndef MECHANICS2DMERGETHINC_NO_MAIN
int main(int argc,char** argv) {
    if(argc<11) {
        std::cerr<<"Usage: "<<argv[0]<<" Nx Ny Lx Ly dt output.e|csv numSteps stride shock_speed_nm_ns cavity_radius_nm [--thinc-beta value] [--density-map file.csv | --riemann]\n"
            <<"Chemistry is enabled. No contact parameters.\n";return 1;
    }
    try {
        Grid g{std::stoi(argv[1]),std::stoi(argv[2]),std::stod(argv[3]),std::stod(argv[4])};
        const double interval=std::stod(argv[5]),speed=std::stod(argv[9]),radius=std::stod(argv[10]);
        const int steps=std::stoi(argv[7]),stride=std::stoi(argv[8]);
        bool riemann=false;std::string densityPath;Settings settings;
        for(int a=11;a<argc;++a) {
            const std::string flag=argv[a];
            if(flag=="--riemann")riemann=true;
            else if(flag=="--density-map" && a+1<argc)densityPath=argv[++a];
            else if(flag=="--thinc-beta" && a+1<argc) {
                std::size_t consumed;const std::string value=argv[++a];
                settings.thincBeta=std::stod(value,&consumed);
                if(consumed!=value.size())throw std::invalid_argument("Invalid THINC beta");
            } else throw std::invalid_argument("Unknown or incomplete option: "+flag);
        }
        if(riemann && !densityPath.empty())throw std::invalid_argument("Density map and Riemann mode are mutually exclusive");
        for(double value:{g.lx,g.ly,interval,speed,radius})if(!std::isfinite(value))throw std::invalid_argument("Inputs must be finite");
        if(g.nx<6 || g.ny<6 || g.lx<=0 || g.ly<=0 || interval<=0 || speed<0 || steps<0 || stride<1)
            throw std::invalid_argument("Invalid grid or time inputs");
        if(!riemann && densityPath.empty() && (radius<=0 || radius>=std::min(.25*g.lx,.5*g.ly)))
            throw std::invalid_argument("Pore radius must be less than min(Lx/4,Ly/2)");
        if(!std::isfinite(settings.thincBeta) || settings.thincBeta<=0 || settings.thincBeta>10)
            throw std::invalid_argument("Require finite 0 < thinc-beta <= 10");
        if(riemann){settings.reactions=false;settings.conduction=false;}
        Eigen::VectorXd mappedDensity;
        if(!densityPath.empty()) {
            const auto map=densityMap2D::read(densityPath);mappedDensity.resize(g.size());
            for(int j=0;j<g.ny;++j)for(int i=0;i<g.nx;++i)
                mappedDensity[g.index(i,j)]=map.at((i+.5)*g.dx(),(j+.5)*g.dy());
            if(mappedDensity.maxCoeff()==0)throw std::invalid_argument("Density map contains no material on FV grid");
            std::cout<<"Loaded intrinsic density map "<<densityPath<<" (nearest neighbor)\n";
        }
        std::cout<<"THINC beta "<<settings.thincBeta<<'\n';
        if(speed>0 && !riemann && densityPath.empty()) {
            const auto shock=downstreamState(speed);settings.driven=true;
            settings.inlet=mixture(1,shock.density,shock.temperature,shock.velocity,0);
            std::cout<<"Shock speed "<<speed<<", particle speed "<<shock.velocity<<", pressure "<<shock.pressure<<" GPa\n";
        }
        if(speed>0 && !densityPath.empty()) {
            settings.driven=true;settings.inletRows.resize(g.ny,Cell::Zero());
            for(int j=0;j<g.ny;++j) {
                const double rho0=mappedDensity[g.index(0,j)];
                if(rho0==0)continue;
                const auto shock=downstreamMappedState(speed,rho0);
                settings.inletRows[j]=mixture(1,shock.density,shock.temperature,shock.velocity,0);
                decode(settings.inletRows[j]);
            }
        }
        // Geometric fraction is used ONCE to initialize the circle, never to
        // remap an evolved conserved state or remove its mass/energy.
        Eigen::VectorXd fractions=Eigen::VectorXd::Ones(g.size());
        if(!riemann && densityPath.empty()) fractions=levelSet2D::geometry(g,levelSet2D::circle(g,g.lx/4,g.ly/2,radius)).fraction;
        State state(8,g.size());
        for(int p=0;p<g.size();++p) {
            double alpha=fractions[p];
            // Remove polygon roundoff only in the initial geometry.
            if(alpha<1e-12)alpha=0;if(alpha>1-1e-12)alpha=1;
            state.col(p)=mixture(alpha,params::rho_init,300,0,0);
            if(!densityPath.empty()) {
                const double rho=mappedDensity[p];
                state.col(p)=rho==0?Cell::Zero():mixture(1,rho,300,0,0);
                try{decode(state.col(p));}catch(const std::exception& e){
                    throw std::invalid_argument("Invalid mapped state at cell ("+std::to_string(p%g.nx)+","+std::to_string(p/g.nx)+"): "+e.what());
                }
            }
            if(riemann) {
                const auto q=riemannMaterial2D::at((p%g.nx+.5)*g.dx()/g.lx,(p/g.nx+.5)*g.dy()/g.ly);
                state.col(p)=mixture(1,q.rhoRatio*params::rho_init,q.T,q.u,q.v);
            }
        }
        const Eigen::Matrix<double,7,1> initial=state.topRows<7>().rowwise().sum()*g.dx()*g.dy();
        Eigen::Matrix<double,7,1> boundary=Eigen::Matrix<double,7,1>::Zero();double reactionHeat=0;
        long internalSteps=0,retries=0,lowOrderSteps=0,faceFallbacks=0;
        const std::vector<std::string> names={"Y1","Y2","Y3","T","rho","rho_material","vx","vy","rhoE","p","phi","material_fraction"};
        std::unique_ptr<ExodusOutput> exodus;std::ofstream csv;
        const std::string path=argv[6],extension=std::filesystem::path(path).extension().string();
        if(extension==".csv") {
            csv.exceptions(std::ios::badbit|std::ios::failbit);csv.open(path);csv<<std::setprecision(17)<<"step,time_ns,X,Y";
            for(const auto& name:names) csv<<','<<name;
            csv<<'\n';
        } else if(extension==".e" || extension==".exo" || extension==".exodus")
            exodus=std::make_unique<ExodusOutput>(path,g.nx,g.ny,g.dx(),g.dy(),names,.5*g.dx(),.5*g.dy());
        else throw std::invalid_argument("Use .csv or .e/.exo/.exodus output");
        std::ofstream diagnostics(path+".diagnostics.csv");diagnostics.exceptions(std::ios::badbit|std::ios::failbit);
        diagnostics<<std::setprecision(17)<<"time_ns,mass,energy,boundary_energy,reaction_heat,mass_error,mx_error,my_error,energy_error,Tmin,Tmax,Tmax_i,Tmax_j,alpha_at_Tmax,rho_at_Tmax,void_area,internal_steps,retries,low_order_steps,face_fallbacks\n";
        auto save=[&](int step,double time) {
            std::vector<Eigen::VectorXd> fields(names.size(),Eigen::VectorXd::Zero(g.size()));
            double Tmin=std::numeric_limits<double>::infinity(),Tmax=0;int hottest=0;
            for(int p=0;p<g.size();++p) {
                fields[4][p]=state(3,p);fields[8][p]=state(6,p);fields[11][p]=state(7,p);
                fields[10][p]=.5-state(7,p); // Interface indicator, NOT signed distance.
                if(state(7,p)==0) continue;
                const auto q=decode(state.col(p));
                for(int k=0;k<3;++k) fields[k][p]=state(k,p)/state(3,p);
                fields[3][p]=q.T;fields[5][p]=q.rho;fields[6][p]=q.u;fields[7][p]=q.v;fields[9][p]=q.p;
                Tmin=std::min(Tmin,q.T);if(q.T>Tmax){Tmax=q.T;hottest=p;}
            }
            if(exodus) exodus->write(time,fields);
            else for(int j=0;j<g.ny;++j) for(int i=0;i<g.nx;++i) {
                const int p=g.index(i,j);csv<<step<<','<<time<<','<<(i+.5)*g.dx()<<','<<(j+.5)*g.dy();
                for(const auto& f:fields) csv<<','<<f[p];
                csv<<'\n';
            }
            const Eigen::Matrix<double,7,1> totals=state.topRows<7>().rowwise().sum()*g.dx()*g.dy();
            Eigen::Matrix<double,7,1> error=totals-initial-boundary;error[6]-=reactionHeat;
            const double voidArea=(g.size()-state.row(7).sum())*g.dx()*g.dy();
            diagnostics<<time<<','<<totals[3]<<','<<totals[6]<<','<<boundary[6]<<','<<reactionHeat<<','<<error[3]<<','<<error[4]<<','<<error[5]<<','<<error[6]
                <<','<<Tmin<<','<<Tmax<<','<<hottest%g.nx<<','<<hottest/g.nx<<','<<state(7,hottest)<<','<<fields[5][hottest]<<','<<voidArea
                <<','<<internalSteps<<','<<retries<<','<<lowOrderSteps<<','<<faceFallbacks<<'\n';diagnostics.flush();
            std::cout<<"time "<<time<<" ns, T=["<<Tmin<<','<<Tmax<<"], void area "<<voidArea
                <<", energy balance error "<<error[6]<<", substeps "<<internalSteps<<", low-order steps "<<lowOrderSteps<<std::endl;
        };
        double time=0;save(0,time);
        for(int step=1;step<=steps;++step) {
            const double target=step*interval;
            while(time<target-1e-12*interval) {
                const auto high=evaluate(state,g,settings,true);
                double dt=std::min(target-time,.9*high.stepLimit);bool accepted=false;std::string failure;
                for(int retry=0;retry<24 && !accepted;++retry) {
                    for(bool order:{true,false}) {
                        try {
                            const auto rate=order?high:evaluate(state,g,settings,false);
                            auto next=RK3Step(state,g,settings,dt,order,rate);
                            state=std::move(next.state);boundary+=next.boundary;reactionHeat+=next.reactionHeat;
                            faceFallbacks+=next.faceFallbacks;++internalSteps;if(!order)++lowOrderSteps;
                            accepted=true;break;
                        } catch(const std::runtime_error& e){failure=e.what();++retries;}
                    }
                    if(!accepted)dt*=.5;
                }
                if(!accepted || time+dt==time) {
                    save(step,time);
                    throw std::runtime_error("Cannot advance at t="+std::to_string(time)+": "+failure);
                }
                time+=dt;
            }
            if(step%stride==0 || step==steps)save(step,time);
        }
        if(exodus)exodus->close();
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
#endif
