#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

// phi > 0 is vacuum; phi <= 0 is material. grad(phi) points into the cavity.
namespace levelSet2D {
struct Grid {
    int nx,ny;
    double lx,ly;
    double dx() const { return lx/nx; }
    double dy() const { return ly/ny; }
    int index(int i,int j) const { return j*nx+i; }
    int size() const { return nx*ny; }
};
struct Point { double x,y,phi; };
struct Geometry {
    Eigen::VectorXd fraction, length, surfaceX, surfaceY, normalX, normalY;
    Eigen::MatrixXd apertureX,apertureY;
};
inline Eigen::VectorXd circle(const Grid& g,double cx,double cy,double radius) {
    if(!(radius>0)) throw std::invalid_argument("Circle radius must be positive");
    Eigen::VectorXd phi(g.size());
    for(int j=0;j<g.ny;++j) for(int i=0;i<g.nx;++i)
        phi[g.index(i,j)]=radius-std::hypot((i+.5)*g.dx()-cx,(j+.5)*g.dy()-cy);
    return phi;
}
inline Point crossing(const Point& a,const Point& b) {
    const double t=a.phi/(a.phi-b.phi);
    return {a.x+t*(b.x-a.x),a.y+t*(b.y-a.y),0};
}
inline double materialLength(double a,double b) {
    if(a<=0 && b<=0) return 1;
    if(a>0 && b>0) return 0;
    const double t=a/(a-b);
    return a<=0 ? t : 1-t;
}
inline double clippedArea(const std::vector<Point>& triangle,std::vector<Point>& intersections) {
    std::vector<Point> polygon;
    for(std::size_t k=0;k<triangle.size();++k) {
        const auto& a=triangle[k]; const auto& b=triangle[(k+1)%triangle.size()];
        if(a.phi<=0) polygon.push_back(a);
        if((a.phi<=0)!=(b.phi<=0)) {
            const auto p=crossing(a,b);polygon.push_back(p);intersections.push_back(p);
        }
    }
    double area=0;
    for(std::size_t k=0;k<polygon.size();++k) {
        const auto& a=polygon[k];const auto& b=polygon[(k+1)%polygon.size()];
        area+=a.x*b.y-a.y*b.x;
    }
    return .5*std::abs(area);
}
inline Geometry geometry(const Grid& g,const Eigen::VectorXd& phi) {
    Geometry out;
    for(auto* f:{&out.fraction,&out.length,&out.surfaceX,&out.surfaceY,&out.normalX,&out.normalY})
        *f=Eigen::VectorXd::Zero(g.size());
    Eigen::MatrixXd corners(g.nx+1,g.ny+1);
    for(int j=0;j<=g.ny;++j) for(int i=0;i<=g.nx;++i) {
        double value=0;
        for(int dj=-1;dj<=0;++dj) for(int di=-1;di<=0;++di)
            value+=phi[g.index(std::clamp(i+di,0,g.nx-1),std::clamp(j+dj,0,g.ny-1))];
        corners(i,j)=value*.25;
    }
    out.apertureX.resize(g.nx+1,g.ny);out.apertureY.resize(g.nx,g.ny+1);
    for(int j=0;j<g.ny;++j) for(int i=0;i<=g.nx;++i)
        out.apertureX(i,j)=materialLength(corners(i,j),corners(i,j+1));
    for(int j=0;j<=g.ny;++j) for(int i=0;i<g.nx;++i)
        out.apertureY(i,j)=materialLength(corners(i,j),corners(i+1,j));
    for(int j=0;j<g.ny;++j) for(int i=0;i<g.nx;++i) {
        const int p=g.index(i,j);
        // Local coordinates keep polygon-area cancellation small on large domains.
        Point a{0,0,corners(i,j)},b{g.dx(),0,corners(i+1,j)},
              c{g.dx(),g.dy(),corners(i+1,j+1)},d{0,g.dy(),corners(i,j+1)};
        double area=0,weightedX=0,weightedY=0;
        for(const auto& tri:{std::vector<Point>{a,b,c},std::vector<Point>{a,c,d}}) {
            std::vector<Point> cuts;
            area+=clippedArea(tri,cuts);
            if(cuts.size()==2) {
                const double length=std::hypot(cuts[1].x-cuts[0].x,cuts[1].y-cuts[0].y);
                out.length[p]+=length;
                weightedX+=length*.5*(cuts[0].x+cuts[1].x);
                weightedY+=length*.5*(cuts[0].y+cuts[1].y);
            }
        }
        out.fraction[p]=std::clamp(area/(g.dx()*g.dy()),0.0,1.0);
        if(out.length[p]>0) {
            out.surfaceX[p]=i*g.dx()+weightedX/out.length[p];
            out.surfaceY[p]=j*g.dy()+weightedY/out.length[p];
            // Normal from material toward vacuum, from the same corner interpolant.
            const double gx=(corners(i+1,j)+corners(i+1,j+1)-corners(i,j)-corners(i,j+1))/(2*g.dx());
            const double gy=(corners(i,j+1)+corners(i+1,j+1)-corners(i,j)-corners(i+1,j))/(2*g.dy());
            const double norm=std::hypot(gx,gy);
            if(norm>0) { out.normalX[p]=gx/norm;out.normalY[p]=gy/norm; }
        }
    }
    return out;
}
inline Eigen::VectorXd transport(const Grid& g,const Eigen::VectorXd& phi,
                                 const Eigen::VectorXd& u,const Eigen::VectorXd& v) {
    Eigen::VectorXd residual(g.size());
    for(int j=0;j<g.ny;++j) for(int i=0;i<g.nx;++i) {
        const int p=g.index(i,j);
        const double gx=u[p]>=0 ? (phi[p]-phi[g.index(std::max(0,i-1),j)])/g.dx()
            : (phi[g.index(std::min(g.nx-1,i+1),j)]-phi[p])/g.dx();
        const double gy=v[p]>=0 ? (phi[p]-phi[g.index(i,std::max(0,j-1))])/g.dy()
            : (phi[g.index(i,std::min(g.ny-1,j+1))]-phi[p])/g.dy();
        residual[p]=-u[p]*gx-v[p]*gy;
    }
    return residual;
}
} // namespace levelSet2D
