// Cartesian cut-cell FV mechanics with a transported vacuum level set.
// Open interfaces have zero traction; paired near-contact patches carry a penalty force.
#include <cmath>
#include <Eigen/Core>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <vector>
#include "modelParams.h"
#include "pressure.h"
#include "computeReactionRates.h"
#include "limiter.h"
#include "levelSet2D.h"
#include "ExodusOutput.h"

using Grid=levelSet2D::Grid;
using Geometry=levelSet2D::Geometry;
// U contains cell-volume-averaged alpha*(rhoY1,rhoY2,rhoY3,rho,mx,my,rhoE).
// Exactly empty cells contain U=0. The material EOS is never evaluated in vacuum.
struct State { Eigen::MatrixXd U; Eigen::VectorXd phi; };
struct Settings {
    double shockSpeed=0,contactStiffness=0,contactDistance=0;
    bool reactions=true;
};
struct Primitive { double rho,T,u,v,p,c; };

// e = cv*T + e_ref(rho,Y1), with e_ref = 0 at rho_init for both phases.
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
    const double a = params::A_unreacted*std::exp(-r1);
    const double b = params::B_unreacted*std::exp(-r2);
    const double slope = rho0*speed*speed;
    const double acousticSlope = a*r1 + b*r2 + rho0*cv*omega*omega*300;
    if (!std::isfinite(slope) || slope <= acousticSlope)
        throw std::invalid_argument("Shock speed must exceed the upstream adiabatic sound speed");

    // With constant cv, de/dV at fixed T is T*dp/dT - p.
    // Integrate the cold pressure and thermal reference term, then impose
    // e2-e1 = (p2+p1)*(V1-V2)/2 to obtain the temperature rise.
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
    // The D-squared terms cancel at the strong-shock compression limit.
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
    if(!std::isfinite(T) || T<=0 || !std::isfinite(p) || !std::isfinite(c2) || c2<0)
        throw std::runtime_error("Invalid material temperature or sound speed");
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

// Conservative cell agglomeration avoids vanishing-volume explicit time steps.
// New/disappearing material cells join a neighboring material cell. Extensive
// quantities are summed and redistributed by new material volume, never clipped.
void mergeSmallCells(State& state,const Grid& g,const Eigen::VectorXd& oldFraction)
{
    const auto geom=levelSet2D::geometry(g,state.phi);
    std::vector<int> parent(g.size());std::iota(parent.begin(),parent.end(),0);
    auto root=[&](int p) { while(parent[p]!=p) {parent[p]=parent[parent[p]];p=parent[p];}return p; };
    for(int j=0;j<g.ny;++j) for(int i=0;i<g.nx;++i) {
        const int p=g.index(i,j);
        const bool occupied=geom.fraction[p]>0 || oldFraction[p]>0 || state.U.col(p).squaredNorm()>0;
        if(!occupied || (geom.fraction[p]>=.25 && oldFraction[p]>=.25)) continue;
        int neighbor=-1;double best=-1;
        for(const auto& offset:{std::pair<int,int>{-1,0},{1,0},{0,-1},{0,1}}) {
            const int a=i+offset.first,b=j+offset.second;
            if(a<0 || a>=g.nx || b<0 || b>=g.ny) continue;
            const int q=g.index(a,b);
            const double score=geom.fraction[q]+oldFraction[q];
            if(geom.fraction[q]>0 && score>best) {best=score;neighbor=q;}
        }
        if(neighbor>=0) parent[root(p)]=root(neighbor);
        else if(geom.fraction[p]==0 && state.U.col(p).norm()>1e-20)
            throw std::runtime_error("Isolated disappearing material cell; refine mesh or reduce dt");
    }
    Eigen::MatrixXd sums=Eigen::MatrixXd::Zero(7,g.size());
    Eigen::VectorXd volumes=Eigen::VectorXd::Zero(g.size());
    for(int p=0;p<g.size();++p) {const int q=root(p);sums.col(q)+=state.U.col(p);volumes[q]+=geom.fraction[p];}
    for(int p=0;p<g.size();++p) {
        const int q=root(p);
        if(volumes[q]>0) state.U.col(p)=sums.col(q)*(geom.fraction[p]/volumes[q]);
        else state.U.col(p).setZero();
    }
}

struct Contact {
    Eigen::MatrixXd force; // Integrated force per unit out-of-plane thickness.
    double potential=0,stepLimit=std::numeric_limits<double>::infinity();
    int pairs=0;
};
Contact contact(const Grid& g,const Geometry& geom,const Eigen::MatrixXd& intrinsic,const Settings& settings)
{
    Contact out{Eigen::MatrixXd::Zero(2,g.size())};
    if(settings.contactStiffness==0 || settings.contactDistance==0) return out;
    std::vector<int> surface;
    for(int p=0;p<g.size();++p) if(geom.length[p]>0 && geom.fraction[p]>0) surface.push_back(p);
    std::vector<int> partner(surface.size(),-1);
    std::vector<double> distance(surface.size(),settings.contactDistance);
    // Only opposing, facing patches qualify. Adjacent patches on the same side
    // cannot repel one another merely because their distance is small.
    for(std::size_t a=0;a<surface.size();++a) for(std::size_t b=a+1;b<surface.size();++b) {
        const int p=surface[a],q=surface[b];
        const double dot=geom.normalX[p]*geom.normalX[q]+geom.normalY[p]*geom.normalY[q];
        if(dot>-.5) continue;
        const double x=geom.surfaceX[q]-geom.surfaceX[p],y=geom.surfaceY[q]-geom.surfaceY[p];
        const double gap=std::hypot(x,y);
        if(gap<=1e-12 || gap>=settings.contactDistance) continue;
        if((x*geom.normalX[p]+y*geom.normalY[p])/gap<.5
           || (x*geom.normalX[q]+y*geom.normalY[q])/gap>-.5) continue;
        // Check that the segment midpoint lies in vacuum rather than through material.
        const int i=std::clamp(int((geom.surfaceX[p]+geom.surfaceX[q])*.5/g.dx()),0,g.nx-1);
        const int j=std::clamp(int((geom.surfaceY[p]+geom.surfaceY[q])*.5/g.dy()),0,g.ny-1);
        if(geom.fraction[g.index(i,j)]>=1) continue;
        if(gap<distance[a]) {distance[a]=gap;partner[a]=int(b);}
        if(gap<distance[b]) {distance[b]=gap;partner[b]=int(a);}
    }
    // Include each patch's closest opposing partner, not only mutual matches.
    // A curved surface can have several fine patches facing one coarse patch.
    // Divide patch lengths among those pairs to avoid multiplying its traction.
    std::set<std::pair<int,int>> pairs;
    std::vector<int> degree(surface.size(),0);
    for(std::size_t a=0;a<surface.size();++a) if(partner[a]>=0)
        pairs.insert(std::minmax(int(a),partner[a]));
    for(const auto& pair:pairs) {++degree[pair.first];++degree[pair.second];}
    for(const auto& pair:pairs) {
        const int a=pair.first,b=pair.second;
        const int p=surface[a],q=surface[b];
        const double gap=std::hypot(geom.surfaceX[q]-geom.surfaceX[p],geom.surfaceY[q]-geom.surfaceY[p]);
        const double overlap=settings.contactDistance-gap;
        const double length=std::min(geom.length[p]/degree[a],geom.length[q]/degree[b]);
        const double magnitude=settings.contactStiffness*overlap*overlap*length;
        const double fx=magnitude*(geom.surfaceX[q]-geom.surfaceX[p])/gap;
        const double fy=magnitude*(geom.surfaceY[q]-geom.surfaceY[p])/gap;
        out.force(0,p)-=fx;out.force(1,p)-=fy;
        out.force(0,q)+=fx;out.force(1,q)+=fy;
        out.potential+=settings.contactStiffness*overlap*overlap*overlap*length/3;
        const double mass=.25*g.dx()*g.dy()*std::min(intrinsic(3,p),intrinsic(3,q));
        out.stepLimit=std::min(out.stepLimit,.1*std::sqrt(mass/(2*settings.contactStiffness*overlap*length)));
        ++out.pairs;
    }
    return out;
}

struct Evaluation {
    Eigen::MatrixXd residual;
    Eigen::VectorXd phiResidual;
    double stepLimit;
};
Evaluation evaluate(const State& state,const Grid& g,const Settings& settings)
{
    const auto geom=levelSet2D::geometry(g,state.phi);
    Eigen::MatrixXd intrinsic=Eigen::MatrixXd::Zero(7,g.size());
    std::vector<Primitive> fields(g.size());
    std::vector<int> owner(g.size(),-1);
    std::queue<int> queue;
    double rate=0;
    for(int p=0;p<g.size();++p) if(geom.fraction[p]>0) {
        intrinsic.col(p)=state.U.col(p)/geom.fraction[p];
        fields[p]=primitive(intrinsic.col(p));owner[p]=p;queue.push(p);
        rate=std::max(rate,(std::abs(fields[p].u)+fields[p].c)/g.dx()
            +(std::abs(fields[p].v)+fields[p].c)/g.dy()
            +2*params::k/(fields[p].rho*params::cv)*(1/(g.dx()*g.dx())+1/(g.dy()*g.dy())));
    }
    if(queue.empty()) throw std::runtime_error("No material remains in the domain");
    // Extend velocity into the void for phi transport only. Vacuum is not assigned
    // material mass/energy; extended states are only used on open grid-face pieces.
    while(!queue.empty()) {
        const int p=queue.front();queue.pop();const int i=p%g.nx,j=p/g.nx;
        for(const auto& offset:{std::pair<int,int>{-1,0},{1,0},{0,-1},{0,1}}) {
            const int a=i+offset.first,b=j+offset.second;
            if(a<0 || a>=g.nx || b<0 || b>=g.ny) continue;
            const int q=g.index(a,b);
            if(owner[q]>=0) continue;
            owner[q]=owner[p];queue.push(q);
        }
    }
    Eigen::VectorXd u(g.size()),v(g.size());
    for(int p=0;p<g.size();++p) {u[p]=fields[owner[p]].u;v[p]=fields[owner[p]].v;}
    Eigen::VectorXd driven;
    Primitive drivenPrimitive{};
    if(settings.shockSpeed>0) {
        const auto shock=downstreamState(settings.shockSpeed);
        driven=materialState(shock.density,shock.temperature,shock.velocity,0);
        drivenPrimitive=primitive(driven);
        rate=std::max(rate,(std::abs(drivenPrimitive.u)+drivenPrimitive.c)/g.dx()+drivenPrimitive.c/g.dy());
    }
    auto numericalFlux=[&](int left,int right,bool alongX,bool inlet) {
        const Eigen::VectorXd UL=inlet ? driven : intrinsic.col(owner[left]).eval();
        const Eigen::VectorXd UR=intrinsic.col(owner[right]);
        const Primitive& a=inlet ? drivenPrimitive : fields[owner[left]];
        const Primitive& b=fields[owner[right]];
        Eigen::VectorXd F=limiter::eval(flux(UL,a,alongX),flux(UR,b,alongX),UL,UR,a.c*a.c,b.c*b.c,
            alongX ? a.u : a.v,alongX ? b.u : b.v);
        F[6]-=params::k*(b.T-a.T)/(alongX ? g.dx() : g.dy());
        return F;
    };
    Evaluation result{Eigen::MatrixXd::Zero(7,g.size()),levelSet2D::transport(g,state.phi,u,v),.1/std::max(rate,1e-30)};
    // Fluxes only cross the material-open portions of grid faces.
    // On the moving phi=0 segment, u_interface=u_material: ALE mass flux is
    // zero, and p_interface=0 gives zero momentum traction and zero pressure work.
    for(int j=0;j<g.ny;++j) for(int face=0;face<=g.nx;++face) {
        const double aperture=geom.apertureX(face,j);if(aperture==0) continue;
        const int left=g.index(std::max(0,face-1),j),right=g.index(std::min(g.nx-1,face),j);
        const Eigen::VectorXd F=aperture*numericalFlux(left,right,true,face==0 && settings.shockSpeed>0)/g.dx();
        if(face>0) result.residual.col(left)-=F;
        if(face<g.nx) result.residual.col(right)+=F;
    }
    for(int face=0;face<=g.ny;++face) for(int i=0;i<g.nx;++i) {
        const double aperture=geom.apertureY(i,face);if(aperture==0) continue;
        const int bottom=g.index(i,std::max(0,face-1)),top=g.index(i,std::min(g.ny-1,face));
        const Eigen::VectorXd F=aperture*numericalFlux(bottom,top,false,false)/g.dy();
        if(face>0) result.residual.col(bottom)-=F;
        if(face<g.ny) result.residual.col(top)+=F;
    }
    const auto forces=contact(g,geom,intrinsic,settings);
    result.stepLimit=std::min(result.stepLimit,forces.stepLimit);
    for(int p=0;p<g.size();++p) if(geom.fraction[p]>0) {
        result.residual(4,p)+=forces.force(0,p)/(g.dx()*g.dy());
        result.residual(5,p)+=forces.force(1,p)/(g.dx()*g.dy());
        result.residual(6,p)+=(forces.force(0,p)*u[p]+forces.force(1,p)*v[p])/(g.dx()*g.dy());
        if(settings.reactions) {
            const double T=fields[p].T,rho=fields[p].rho;
            const double r1=reactionRates::computeArrhenius(T,params::R,params::E1,params::Z1);
            const double r2=reactionRates::computeArrhenius(T,params::R,params::E2,params::Z2);
            result.residual(0,p)-=r1*state.U(0,p);
            result.residual(1,p)+=r1*state.U(0,p)-r2*state.U(1,p);
            result.residual(2,p)+=r2*state.U(1,p);
            result.residual(6,p)+=(params::rho_init/rho)
                *(reactionRates::computeReactionHeat(params::a1,params::b1,T)*r1*state.U(0,p)
                +reactionRates::computeReactionHeat(params::a2,params::b2,T)*r2*state.U(1,p));
            result.stepLimit=std::min(result.stepLimit,.1/std::max({r1,r2,1e-30}));
        }
    }
    return result;
}

void validate(const State& state,const Grid& g)
{
    if(!state.U.allFinite() || !state.phi.allFinite()) throw std::runtime_error("Nonfinite solution");
    const auto geom=levelSet2D::geometry(g,state.phi);
    for(int p=0;p<g.size();++p) if(geom.fraction[p]>0) primitive(state.U.col(p)/geom.fraction[p]);
}
State forwardEuler(const State& state,const Grid& g,double dt,const Evaluation& rate)
{
    if(dt>rate.stepLimit*(1+1e-12)) throw std::runtime_error("Stage timestep bound exceeded");
    State next{state.U+dt*rate.residual,state.phi+dt*rate.phiResidual};
    mergeSmallCells(next,g,levelSet2D::geometry(g,state.phi).fraction);
    validate(next,g);return next;
}
State RK3Step(const State& state,const Grid& g,const Settings& settings,double dt,const Evaluation& rate)
{
    auto a=forwardEuler(state,g,dt,rate);
    auto b=forwardEuler(a,g,dt,evaluate(a,g,settings));
    b.U=.75*state.U+.25*b.U;b.phi=.75*state.phi+.25*b.phi;
    mergeSmallCells(b,g,levelSet2D::geometry(g,a.phi).fraction);validate(b,g);
    auto c=forwardEuler(b,g,dt,evaluate(b,g,settings));
    c.U=(1.0/3)*state.U+(2.0/3)*c.U;c.phi=(1.0/3)*state.phi+(2.0/3)*c.phi;
    mergeSmallCells(c,g,levelSet2D::geometry(g,b.phi).fraction);validate(c,g);
    return c;
}

#ifndef MECHANICS2DLEVELSET_NO_MAIN
int main(int argc,char** argv)
{
    if(argc<13 || argc>14) {
        std::cerr << "Usage: " << argv[0]
            << " Nx Ny Lx Ly dt output.e|csv numSteps stride shock_speed_nm_ns cavity_radius_nm contact_K_GPa_per_nm2 contact_distance_nm [--no-reactions]\n";
        return 1;
    }
    try {
        Grid g{std::stoi(argv[1]),std::stoi(argv[2]),std::stod(argv[3]),std::stod(argv[4])};
        const double interval=std::stod(argv[5]),radius=std::stod(argv[10]);
        const int steps=std::stoi(argv[7]),stride=std::stoi(argv[8]);
        Settings settings{std::stod(argv[9]),std::stod(argv[11]),std::stod(argv[12]),true};
        if(argc==14) {
            if(std::string(argv[13])!="--no-reactions") throw std::invalid_argument("Expected --no-reactions");
            settings.reactions=false;
        }
        for(double value:{g.lx,g.ly,interval,radius,settings.shockSpeed,settings.contactStiffness,settings.contactDistance})
            if(!std::isfinite(value)) throw std::invalid_argument("Inputs must be finite");
        if(g.nx<4 || g.ny<4 || g.lx<=0 || g.ly<=0 || interval<=0 || radius<=0
            || radius>=.5*std::min(g.lx,g.ly) || steps<0 || stride<1 || settings.shockSpeed<0
            || settings.contactStiffness<0 || settings.contactDistance<0)
            throw std::invalid_argument("Invalid grid, radius, loading, contact or output inputs");
        if(settings.contactStiffness>0 && settings.contactDistance<2*std::max(g.dx(),g.dy()))
            throw std::invalid_argument("Contact distance must span at least two grid cells");
        if(settings.shockSpeed>0) {
            const auto shock=downstreamState(settings.shockSpeed);
            std::cout << "Shock speed " << settings.shockSpeed << " nm/ns, particle velocity " << shock.velocity
                << " nm/ns, pressure " << shock.pressure << " GPa, temperature " << shock.temperature << " K\n";
        }
        State state{Eigen::MatrixXd::Zero(7,g.size()),levelSet2D::circle(g,g.lx/2,g.ly/2,radius)};
        const auto initialGeometry=levelSet2D::geometry(g,state.phi);
        const auto initial=materialState(params::rho_init,300,0,0);
        for(int p=0;p<g.size();++p) state.U.col(p)=initialGeometry.fraction[p]*initial;
        const std::vector<std::string> names={"Y1","Y2","Y3","T","rho","rho_material","vx","vy","rhoE","p","phi","material_fraction","contact_fx","contact_fy"};
        std::ofstream csv;
        std::unique_ptr<ExodusOutput> exodus;
        const std::string extension=std::filesystem::path(argv[6]).extension().string();
        if(extension==".csv") {
            csv.exceptions(std::ios::badbit|std::ios::failbit);csv.open(argv[6]);
            csv << std::setprecision(17) << "step,time_ns,X,Y";
            for(const auto& name:names) csv << ',' << name;csv << '\n';
        } else if(extension==".e" || extension==".exo" || extension==".exodus")
            exodus=std::make_unique<ExodusOutput>(argv[6],g.nx,g.ny,g.dx(),g.dy(),names,.5*g.dx(),.5*g.dy());
        else throw std::invalid_argument("Use .csv, .e, .exo, or .exodus");
        auto save=[&](int step,double time) {
            const auto geom=levelSet2D::geometry(g,state.phi);
            std::vector<Eigen::VectorXd> fields(names.size(),Eigen::VectorXd::Zero(g.size()));
            Eigen::MatrixXd intrinsic=Eigen::MatrixXd::Zero(7,g.size());
            for(int p=0;p<g.size();++p) {
                fields[10][p]=state.phi[p];fields[11][p]=geom.fraction[p];
                fields[4][p]=state.U(3,p);fields[8][p]=state.U(6,p);
                if(geom.fraction[p]==0) continue;
                intrinsic.col(p)=state.U.col(p)/geom.fraction[p];const auto q=primitive(intrinsic.col(p));
                for(int k=0;k<3;++k) fields[k][p]=intrinsic(k,p)/q.rho;
                fields[3][p]=q.T;fields[5][p]=q.rho;fields[6][p]=q.u;fields[7][p]=q.v;fields[9][p]=q.p;
            }
            const auto forces=contact(g,geom,intrinsic,settings);
            fields[12]=forces.force.row(0).transpose()/(g.dx()*g.dy());
            fields[13]=forces.force.row(1).transpose()/(g.dx()*g.dy());
            if(exodus) exodus->write(time,fields);
            else for(int j=0;j<g.ny;++j) for(int i=0;i<g.nx;++i) {
                const int p=g.index(i,j);csv << step << ',' << time << ',' << (i+.5)*g.dx() << ',' << (j+.5)*g.dy();
                for(const auto& field:fields) csv << ',' << field[p];csv << '\n';
            }
            std::cout << "time " << time << " ns, contact pairs " << forces.pairs
                << ", material mass " << state.U.row(3).sum()*g.dx()*g.dy() << '\n';
        };
        double time=0;save(0,time);
        for(int step=1;step<=steps;++step) {
            const double target=step*interval;
            while(time<target-1e-13*interval) {
                const auto rate=evaluate(state,g,settings);
                double dt=std::min(target-time,rate.stepLimit*.9);
                bool accepted=false;
                for(int retry=0;retry<24;++retry) {
                    try {auto candidate=RK3Step(state,g,settings,dt,rate);state=std::move(candidate);accepted=true;break;}
                    catch(const std::runtime_error&) {dt*=.5;}
                }
                if(!accepted || dt<1e-14*interval) throw std::runtime_error("Cannot advance an admissible interface state; refine the mesh or revise contact parameters");
                time+=dt;
            }
            if(step%stride==0 || step==steps) save(step,time);
        }
        if(exodus) exodus->close();
    } catch(const std::exception& error) {std::cerr << error.what() << '\n';return 1;}
}
#endif
