#define MECHANICS2DLEVELSET_NO_MAIN
#include "../src/mechanics2DLevelSet.C"
#include <cassert>

void near(double a,double b,double tolerance=1e-10) {
    if(!std::isfinite(a) || !std::isfinite(b) || std::abs(a-b)>tolerance*std::max({1.0,std::abs(a),std::abs(b)}))
        throw std::runtime_error("Level-set comparison failed");
}
State filled(const Grid& g,const Eigen::VectorXd& phi,double T=300) {
    State s{Eigen::MatrixXd::Zero(7,g.size()),phi};
    const auto geom=levelSet2D::geometry(g,phi);
    for(int p=0;p<g.size();++p) s.U.col(p)=geom.fraction[p]*materialState(params::rho_init,T,0,0);
    return s;
}
int main() {
    Grid g{60,60,120,120};
    auto phi=levelSet2D::circle(g,60,60,20);
    auto geom=levelSet2D::geometry(g,phi);
    const double area=(Eigen::VectorXd::Ones(g.size())-geom.fraction).sum()*g.dx()*g.dy();
    near(area,std::acos(-1.0)*400,.01);
    for(int p=0;p<g.size();++p) if(geom.length[p]>0) {
        const double x=60-geom.surfaceX[p],y=60-geom.surfaceY[p];
        assert((geom.normalX[p]*x+geom.normalY[p]*y)/std::hypot(x,y)>.99);
    }
    // Exact derivative of a linear level set under uniform translation, away from boundaries.
    Eigen::VectorXd linear(g.size()),u=Eigen::VectorXd::Constant(g.size(),2),v=Eigen::VectorXd::Constant(g.size(),-3);
    for(int j=0;j<g.ny;++j) for(int i=0;i<g.nx;++i) linear[g.index(i,j)]=(i+.5)*g.dx()+2*(j+.5)*g.dy();
    auto advected=levelSet2D::transport(g,linear,u,v);
    for(int j=1;j<g.ny-1;++j) for(int i=1;i<g.nx-1;++i) near(advected[g.index(i,j)],4);

    Settings settings;settings.reactions=false;
    auto s=filled(g,phi);auto rate=evaluate(s,g,settings);
    near(rate.residual.norm(),0,1e-8);near(rate.phiResidual.norm(),0);
    auto next=RK3Step(s,g,settings,1e-6,rate);
    near((next.U-s.U).norm(),0,1e-8);near((next.phi-s.phi).norm(),0,1e-8);
    const auto before=s.U.rowwise().sum().eval();
    s.phi=levelSet2D::circle(g,60.1,60,20);
    mergeSmallCells(s,g,geom.fraction);
    for(int k=0;k<7;++k) near(s.U.row(k).sum(),before[k]);

    // Uniform positive material pressure pushes toward the cavity; there is no
    // opposing interface pressure/traction. No energy flux exists at rest.
    s=filled(g,phi,400);rate=evaluate(s,g,settings);
    near(rate.residual.row(6).norm(),0,1e-8);
    double accelerationIntoVacuum=0;
    for(int p=0;p<g.size();++p) if(geom.length[p]>0)
        accelerationIntoVacuum+=rate.residual(4,p)*geom.normalX[p]+rate.residual(5,p)*geom.normalY[p];
    assert(accelerationIntoVacuum>0);

    // Two facing flat surfaces: repulsive forces are equal/opposite and vanish
    // once their gap exceeds the prescribed finite contact layer.
    for(int j=0;j<g.ny;++j) for(int i=0;i<g.nx;++i) phi[g.index(i,j)]=5-std::abs((i+.5)*g.dx()-60);
    geom=levelSet2D::geometry(g,phi);
    Eigen::MatrixXd intrinsic(7,g.size());
    for(int p=0;p<g.size();++p) intrinsic.col(p)=materialState(params::rho_init,300,0,0);
    settings.contactStiffness=.01;settings.contactDistance=15;
    const auto forces=contact(g,geom,intrinsic,settings);
    assert(forces.pairs>0 && forces.potential>0);
    near(forces.force.row(0).sum(),0);near(forces.force.row(1).sum(),0);
    for(int p=0;p<g.size();++p) if(forces.force.col(p).norm()>0)
        assert(forces.force(0,p)*geom.normalX[p]+forces.force(1,p)*geom.normalY[p]<0);
    settings.contactDistance=9;
    assert(contact(g,geom,intrinsic,settings).pairs==0);
    std::cout << "Passed: circular geometry/normals, level-set transport, stationary vacuum, conservative agglomeration, free-surface pressure response, paired repulsion and contact cutoff.\n";
}
