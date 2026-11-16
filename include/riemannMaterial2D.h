#pragma once
// Four-quadrant test for the EXISTING material EOS (not an ideal-gas benchmark).
// Coordinates below are fractions of the domain. T: K; velocities: nm/ns.
// All states are pure unreacted material. --riemann disables chemistry and
// conduction and uses zero-gradient outflow on all four boundaries.
namespace riemannMaterial2D {
struct State { double rhoRatio, T, u, v; };
inline constexpr double splitX=.5, splitY=.5;
// Upper right, upper left, lower left, lower right.
inline constexpr State quadrants[] = {
    {1.0,300,0,0}, {1.0,400,100,0},
    {1.0,500,100,100}, {1.0,400,0,100}
};
inline State at(double x,double y) {
    return quadrants[y>=splitY ? (x>=splitX?0:1) : (x>=splitX?3:2)];
}
}
