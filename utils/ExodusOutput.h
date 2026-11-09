#pragma once
#include <exodusII.h>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// One static QUAD4 mesh with four double-precision nodal time histories.
class ExodusOutput {
    int id_ = -1;
    int snapshot_ = 0;
    int nodes_ = 0;
    static void check(int code, const char* operation) {
        if (code < 0) throw std::runtime_error(std::string("Exodus: ") + operation);
    }
public:
    ExodusOutput(const std::string& path, int nx, int ny, double dx, double dy) {
        const long long nodes = 1LL * nx * ny;
        const long long cells = 1LL * (nx-1) * (ny-1);
        if (nodes > std::numeric_limits<int>::max() || cells > std::numeric_limits<int>::max())
            throw std::invalid_argument("Exodus writer requires node/element counts within 32-bit limits");
        nodes_ = static_cast<int>(nodes);
        int cpuWord = sizeof(double), fileWord = sizeof(double);
        // NetCDF 64-bit-offset format; avoids requiring HDF5 at build time.
        id_ = ex_create(path.c_str(), EX_CLOBBER | EX_LARGE_MODEL, &cpuWord, &fileWord);
        if (id_ < 0) throw std::runtime_error("Cannot create Exodus file: " + path);
        try {
            check(ex_put_init(id_, "2D reaction diffusion; length nm, time ns, temperature K",
                              2, nodes, cells, 1, 0, 0), "initialize mesh");
            std::vector<double> x(nodes_), y(nodes_);
            for (int j=0; j<ny; ++j) for (int i=0; i<nx; ++i) {
                const int p=j*nx+i;
                x[p]=i*dx; y[p]=j*dy;
            }
            check(ex_put_coord(id_, x.data(), y.data(), nullptr), "write coordinates");
            char xname[]="X", yname[]="Y";
            char* coords[]={xname,yname};
            check(ex_put_coord_names(id_, coords), "name coordinates");
            check(ex_put_block(id_, EX_ELEM_BLOCK, 1, "QUAD4", cells, 4, 0, 0, 0), "define quadrilateral block");
            check(ex_put_name(id_, EX_ELEM_BLOCK, 1, "material"), "name block");
            std::vector<int> conn;
            conn.reserve(static_cast<std::size_t>(cells)*4);
            for (int j=0; j<ny-1; ++j) for (int i=0; i<nx-1; ++i) {
                const int p=j*nx+i+1; // Exodus connectivity is one-based, counterclockwise.
                conn.insert(conn.end(), {p,p+1,p+nx+1,p+nx});
            }
            check(ex_put_conn(id_, EX_ELEM_BLOCK, 1, conn.data(), nullptr, nullptr), "write connectivity");
            check(ex_put_variable_param(id_, EX_NODAL, 4), "define nodal fields");
            char a[]="Y1", b[]="Y2", c[]="Y3", d[]="T";
            char* names[]={a,b,c,d};
            check(ex_put_variable_names(id_, EX_NODAL, 4, names), "name nodal fields");
        } catch (...) {
            ex_close(id_); id_=-1; throw;
        }
    }
    ExodusOutput(const ExodusOutput&) = delete;
    ExodusOutput& operator=(const ExodusOutput&) = delete;
    ~ExodusOutput() { if (id_>=0) ex_close(id_); }
    template<class State>
    void write(double time, const State& state) {
        ++snapshot_;
        check(ex_put_time(id_, snapshot_, &time), "write time");
        for (int k=0; k<4; ++k) {
            if (state[k].size()!=nodes_) throw std::runtime_error("Exodus field size does not match mesh");
            check(ex_put_var(id_, snapshot_, EX_NODAL, k+1, 1, nodes_, state[k].data()), "write nodal field");
        }
        check(ex_update(id_), "flush snapshot");
    }
    void close() {
        const int id=id_; id_=-1;
        if (id>=0) check(ex_close(id), "close file");
    }
};
