// 2D nodal finite differences and SSP RK3, with insulated boundaries.
#include <Eigen/Core>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <filesystem>
#include <memory>
#include "ExodusOutput.h"
#include "modelParams.h"
#include "computeReactionRates.h"

namespace solver2d {
using State = std::array<Eigen::VectorXd, 4>;
struct Grid {
    int nx, ny;
    double lx, ly;
    double dx() const { return lx / (nx - 1); }
    double dy() const { return ly / (ny - 1); }
    Eigen::Index size() const { return Eigen::Index(nx) * ny; }
    Eigen::Index index(int i, int j) const { return Eigen::Index(j) * nx + i; }
};
State zeros(const Grid& g) {
    State u;
    for (auto& field : u) field = Eigen::VectorXd::Zero(g.size());
    return u;
}
State initialState(const Grid& g, double hot, double cold) {
    State u = zeros(g);
    u[0].setConstant(params::Y1_init);
    u[1].setConstant(params::Y2_init);
    u[2].setConstant(params::Y3_init);
    u[3].setConstant(cold);
    // Central 20% in each direction; use physical coordinates for refinement.

    //add circular initial condition
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
            if (std::pow(i * g.dx() - 0.5 * g.lx, 2) + std::pow(j * g.dy() - 0.5 * g.ly, 2) <= std::pow(0.1 * g.lx, 2))
                u[3][g.index(i,j)] = hot;
    return u;
}
State rhs(const State& u, const Grid& g) {
    State r = zeros(g);
    const double ax = params::alpha / (g.dx() * g.dx());
    const double ay = params::alpha / (g.dy() * g.dy());
    for (int j = 0; j < g.ny; ++j) {
        // Mirrored ghost nodes impose dT/dn = 0 at both y boundaries.
        const int jm = j == 0 ? 1 : j - 1;
        const int jp = j == g.ny - 1 ? g.ny - 2 : j + 1;
        for (int i = 0; i < g.nx; ++i) {
            const int im = i == 0 ? 1 : i - 1;
            const int ip = i == g.nx - 1 ? g.nx - 2 : i + 1;
            const auto p = g.index(i,j);
            const double t = u[3][p];
            if (!std::isfinite(t) || t <= 0)
                throw std::runtime_error("Nonpositive/nonfinite stage temperature; reduce dt or check parameters");
            const double r1 = reactionRates::computeArrhenius(t, params::R, params::E1, params::Z1);
            const double r2 = reactionRates::computeArrhenius(t, params::R, params::E2, params::Z2);
            const double w1 = r1 * u[0][p], w2 = r2 * u[1][p];
            const double q1 = reactionRates::computeReactionHeat(params::a1, params::b1, t);
            const double q2 = reactionRates::computeReactionHeat(params::a2, params::b2, t);
            r[0][p] = -w1;
            r[1][p] = w1 - w2;
            r[2][p] = w2;
            // Add both directions and chemistry at every node, including corners.
            r[3][p] = (q1 * w1 + q2 * w2) / params::cv
                + ax * (u[3][g.index(im,j)] - 2*t + u[3][g.index(ip,j)])
                + ay * (u[3][g.index(i,jm)] - 2*t + u[3][g.index(i,jp)]);
        }
    }
    return r;
}
void step(State& u, const Grid& g, double dt) {
    const State old = u;
    State u1 = old, u2 = old;
    auto r = rhs(old,g);
    for (int k=0; k<4; ++k) u1[k] = old[k] + dt*r[k];
    r = rhs(u1,g);
    for (int k=0; k<4; ++k) u2[k] = .75*old[k] + .25*u1[k] + .25*dt*r[k];
    r = rhs(u2,g);
    for (int k=0; k<4; ++k) {
        u[k] = (1.0/3)*old[k] + (2.0/3)*u2[k] + (2.0/3)*dt*r[k];
        if (!u[k].allFinite()) throw std::runtime_error("Nonfinite solution; reduce dt or check parameters");
    }
}
void write(std::ostream& out, const State& u, const Grid& g, int completed, double dt) {
    // Match 1D's initial snapshot label 1, but also record physical time explicitly.
    for (int j=0; j<g.ny; ++j)
        for (int i=0; i<g.nx; ++i) {
            const auto p = g.index(i,j);
            out << completed+1 << ',' << completed*dt << ',' << i*g.dx() << ',' << j*g.dy();
            for (int k=0; k<4; ++k) out << ',' << u[k][p];
            out << '\n';
        }
    if (!out) throw std::runtime_error("Failed to write output");
}
int integer(const char* arg) {
    std::size_t end;
    int value = std::stoi(arg,&end);
    if (arg[end] != '\0') throw std::invalid_argument("Expected an integer");
    return value;
}
double positive(const char* arg) {
    std::size_t end;
    double value = std::stod(arg,&end);
    if (arg[end] != '\0' || !std::isfinite(value) || value <= 0)
        throw std::invalid_argument("Expected a finite positive number");
    return value;
}
} // namespace solver2d

#ifndef SOLVER2D_NO_MAIN
int main(int argc, char** argv) {
    using namespace solver2d;
    if (argc != 11) {
        std::cerr << "Usage: " << argv[0]
                  << " Nx Ny Lx Ly T_init T_0 dt output.exodus numSteps stride\n";
        return 1;
    }
    try {
        Grid g{integer(argv[1]), integer(argv[2]), positive(argv[3]), positive(argv[4])};
        const double hot=positive(argv[5]), cold=positive(argv[6]), dt=positive(argv[7]);
        const int count=integer(argv[9]), stride=integer(argv[10]);
        if (g.nx<3 || g.ny<3 || hot<cold || count<0 || count==std::numeric_limits<int>::max() || stride<=0)
            throw std::invalid_argument("Require Nx,Ny >= 3, T_init >= T_0, numSteps >= 0, stride > 0");
        const double diffusionNumber = params::alpha*dt*(1/(g.dx()*g.dx())+1/(g.dy()*g.dy()));
        if (!std::isfinite(diffusionNumber) || diffusionNumber > .5)
            throw std::invalid_argument("Diffusion SSP bound exceeded: alpha*dt*(1/dx^2+1/dy^2) must be <= 0.5; chemistry may require smaller dt");
        auto u=initialState(g,hot,cold);
        const std::string extension=std::filesystem::path(argv[8]).extension().string();
        const bool csv=extension==".csv";
        if (!csv && extension!=".e" && extension!=".exo" && extension!=".exodus")
            throw std::invalid_argument("Output extension must be .exodus, .exo, .e, or .csv");
        std::ofstream out;
        std::unique_ptr<ExodusOutput> exodus;
        if (csv) {
            out.open(argv[8]);
            if (!out) throw std::runtime_error("Cannot open output file");
            out << std::setprecision(17) << "step,time_ns,X,Y,Y1,Y2,Y3,T\n";
        } else {
            exodus=std::make_unique<ExodusOutput>(argv[8],g.nx,g.ny,g.dx(),g.dy());
        }
        auto save = [&](int completed) {
            if (exodus) exodus->write(completed*dt,u);
            else write(out,u,g,completed,dt);
        };
        save(0);
        for (int n=1; n<=count; ++n) {
            step(u,g,dt);
            if (n%stride==0 || n==count) save(n);
        }
        if (exodus) exodus->close();
        std::cout << "Completed " << count << " steps on " << g.nx << 'x' << g.ny << " nodes\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
#endif
