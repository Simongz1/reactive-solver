// Measure angular variation of a contour on a rectangular Exodus nodal grid.
#include <exodusII.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

void check(int result) {
    if(result<0) throw std::runtime_error("Exodus read failed");
}

int main(int argc,char** argv)
{
    if(argc!=4) {
        std::cerr << "Usage: " << argv[0] << " output.e field threshold\n";
        return 1;
    }
    int id=-1;
    try {
        const double threshold=std::stod(argv[3]);
        if(!std::isfinite(threshold)) throw std::runtime_error("Threshold must be finite");
        int cpu=8, disk=0; float version;
        id=ex_open(argv[1],EX_READ,&cpu,&disk,&version); check(id);
        const int nodes=ex_inquire_int(id,EX_INQ_NODES), steps=ex_inquire_int(id,EX_INQ_TIME);
        int fields; check(ex_get_variable_param(id,EX_NODAL,&fields));
        std::vector<std::vector<char>> labels(fields,std::vector<char>(MAX_STR_LENGTH+1));
        std::vector<char*> names;
        for(auto& label:labels) names.push_back(label.data());
        check(ex_get_variable_names(id,EX_NODAL,fields,names.data()));
        int field=-1;
        for(int k=0;k<fields;++k) if(std::string(names[k])==argv[2]) field=k+1;
        if(field<0) throw std::runtime_error("Field not found");
        std::vector<double> x(nodes),y(nodes),time(steps),values(nodes);
        check(ex_get_coord(id,x.data(),y.data(),nullptr));
        check(ex_get_all_times(id,time.data()));
        int nx=1;
        while(nx<nodes && y[nx]==y[0]) ++nx;
        if(nx<2 || nodes%nx || nodes/nx<2) throw std::runtime_error("Expected an ordered rectangular grid");
        const int ny=nodes/nx;
        const double dx=x[1]-x[0],dy=y[nx]-y[0];
        if(dx<=0 || dy<=0) throw std::runtime_error("Expected increasing grid coordinates");
        for(int j=0;j<ny;++j) for(int i=0;i<nx;++i)
            if(std::abs(x[j*nx+i]-(x[0]+i*dx))>1e-8*nx*dx
               || std::abs(y[j*nx+i]-(y[0]+j*dy))>1e-8*ny*dy)
                throw std::runtime_error("Grid must be uniform and ordered by y then x");
        const double cx=.5*(x[0]+x[nx-1]),cy=.5*(y[0]+y[nodes-nx]);
        const double dr=std::min(dx,dy)/5;
        const double limit=.999*std::min(cx-x[0],cy-y[0]);
        auto sample=[&](double px,double py) {
            const double ix=(px-x[0])/dx, iy=(py-y[0])/dy;
            const int i=std::clamp(int(ix),0,nx-2),j=std::clamp(int(iy),0,ny-2);
            const double a=ix-i,b=iy-j;
            return (1-b)*((1-a)*values[j*nx+i]+a*values[j*nx+i+1])
                + b*((1-a)*values[(j+1)*nx+i]+a*values[(j+1)*nx+i+1]);
        };
        std::cout << std::setprecision(12)
            << "time_ns,r_axis_nm,r_diagonal_nm,r_min_nm,r_max_nm,relative_spread,valid_rays\n";
        for(int step=1;step<=steps;++step) {
            check(ex_get_var(id,step,EX_NODAL,field,1,nodes,values.data()));
            std::vector<double> radii(64,std::numeric_limits<double>::quiet_NaN());
            double total=0,minimum=std::numeric_limits<double>::infinity(),maximum=0;
            int valid=0;
            for(int angle=0;angle<64;++angle) {
                const double theta=angle*2*std::acos(-1.0)/64;
                double previous=sample(cx,cy),radius=std::numeric_limits<double>::quiet_NaN();
                for(double r=dr;r<limit;r+=dr) {
                    const double value=sample(cx+r*std::cos(theta),cy+r*std::sin(theta));
                    if(previous>=threshold && value<threshold)
                        radius=r-dr+dr*(previous-threshold)/(previous-value);
                    previous=value;
                }
                // Exclude rays whose outer contour is outside the inscribed disk.
                if(previous>=threshold) radius=std::numeric_limits<double>::quiet_NaN();
                radii[angle]=radius;
                if(std::isfinite(radius)) {
                    ++valid; total+=radius;
                    minimum=std::min(minimum,radius); maximum=std::max(maximum,radius);
                }
            }
            const double nan=std::numeric_limits<double>::quiet_NaN();
            const double spread=valid==64 && total>0 ? (maximum-minimum)/(total/valid) : nan;
            std::cout << time[step-1] << ',' << radii[0] << ',' << radii[8] << ','
                << (valid ? minimum : nan) << ',' << (valid ? maximum : nan) << ',' << spread << ',' << valid << '\n';
        }
        check(ex_close(id)); id=-1;
    } catch(const std::exception& error) {
        if(id>=0) ex_close(id);
        std::cerr << error.what() << '\n'; return 1;
    }
}
