#define MECHANICS2DMULTIMATERIAL_NO_MAIN
#include "../src/mechanics2DMultimaterial.C"
#include <cassert>
void near(double a, double b, double relative = 1e-10, double absolute = 1e-14)
{
    static int comparison=0; ++comparison;
    if (std::abs(a - b) > absolute + relative * std::max(std::abs(a), std::abs(b)))
        throw std::runtime_error("Mismatch #"+std::to_string(comparison)+": " + std::to_string(a) + " vs " + std::to_string(b));
}
int main()
{
    const std::array<double, 3> densities{params::rho_init, multimaterialEOS::rhoPS, 1.2e-9};
    for (int phase = 0; phase < 3; ++phase)
    {
        auto U = pureState(phase, densities[phase], 700, 123);
        auto q = decode(U);
        near(q.T, 700);
        near(q.u, 123);
        near(q.rho, densities[phase]);
        near(q.p, multimaterialEOS::pressureValue(phase, densities[phase], 700));
        if (phase != 0)
            near(U.head<3>().sum(), 0);
        // Check reference energy derivative against the thermodynamic identity.
        const double rho = densities[phase] * 1.02, h = rho * 1e-5;
        const double derivative = (multimaterialEOS::referenceEnergy(phase, rho + h) -
                                   multimaterialEOS::referenceEnergy(phase, rho - h)) /
                                  (2 * h);
        near(derivative * rho * rho,
             (multimaterialEOS::pressureValue(phase, rho, 700) -
              700 * multimaterialEOS::pressureTemperatureDerivative(phase, rho, 700)),
             1e-7, 1e-10);
    }
    // A phase's mass can underflow before its subnormal volume fraction.
    Cell tail = pureState(0, densities[0]);
    tail[AlphaPore] = 2.1304407433908008e-315;
    const auto tailState = decode(tail);
    near(tailState.T, 300);
    near(tailState.p, decode(pureState(0, densities[0])).p);
    // Finite mass/fraction inconsistency must still fail, not be clipped.
    tail[AlphaPore] = 1e-8;
    bool rejected = false;
    try { decode(tail); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);

    const Cell mixed = makeState({.3, .4, .3}, densities, 900, 42, -17, {.2, .3, .5});
    auto q = decode(mixed);
    near(q.T, 900);
    near(q.species[0], .2);
    double expected = 0;
    for (int k = 0; k < 3; ++k)
        expected += q.alpha[k] * multimaterialEOS::pressureValue(k, densities[k], 900, .2);
    near(q.p, expected);
    Grid grid{12, 10, 120, 100};
    Settings settings;
    settings.conduction = false;
    State state(stateFieldCount, grid.size());
    for (int cell = 0; cell < grid.size(); ++cell)
        state.col(cell) = mixed;
    const auto rate = evaluate(state, grid, settings);
    const double dt = std::min(1e-6, .5 * rate.stepLimit);
    const auto next = RK3Step(state, grid, settings, dt, true, rate);
    const Conserved difference =
        (next.state.topRows<conservedFieldCount>() - state.topRows<conservedFieldCount>())
                .rowwise()
                .sum() *
            grid.dx() * grid.dy() -
        next.boundary;
    for (int k = AlphaRhoRDX; k < TotalEnergy; ++k)
        near(difference[k], 0, 1e-8, 1e-10);
    near(difference[TotalEnergy], next.reactionHeat, 1e-6, 1e-8);
    // No chemistry in pure polymer or air, even at high temperature.
    for (int phase : {1, 2})
    {
        for (int cell = 0; cell < grid.size(); ++cell)
            state.col(cell) = pureState(phase, densities[phase], 2000);
        auto r = evaluate(state, grid, settings);
        near(r.rhs.norm(), 0);
        near(r.reactionHeat, 0);
    }
    // A moving interface: conservative partial-mass transport and fraction sum.
    settings.reactions = false;
    for (int j = 0; j < grid.ny; ++j)
        for (int i = 0; i < grid.nx; ++i)
            state.col(grid.index(i, j)) = pureState(i < 4   ? 0
                                                    : i < 8 ? 1
                                                            : 2,
                                                    densities[i < 4   ? 0
                                                              : i < 8 ? 1
                                                                      : 2],
                                                    300, 100);
    const Conserved initial =
        state.topRows<conservedFieldCount>().rowwise().sum() * grid.dx() * grid.dy();
    Conserved boundary = Conserved::Zero();
    for (int n = 0; n < 10; ++n)
    {
        auto r = evaluate(state, grid, settings);
        auto step = RK3Step(state, grid, settings, std::min(1e-7, .1 * r.stepLimit), false,
                            evaluate(state, grid, settings, false));
        state = step.state;
        boundary += step.boundary;
    }
    Conserved error = state.topRows<conservedFieldCount>().rowwise().sum() * grid.dx() * grid.dy() -
                      initial - boundary;
    for (int k = AlphaRhoRDX; k <= TotalEnergy; ++k)
        near(error[k], 0, 1e-8, 1e-8);
    for (int cell = 0; cell < grid.size(); ++cell)
        near(state.col(cell).tail<3>().sum(), 1);
    // A uniform state moving with the mesh/wall remains uniform for every EOS.
    settings.pistonVelocity=100;
    for(int phase=0;phase<3;++phase) {
        for(int cell=0;cell<grid.size();++cell)
            state.col(cell)=pureState(phase,densities[phase],300,100);
        auto r=evaluate(state,grid,settings);
        near(r.rhs.norm(),0,1e-8,1e-10);
    }
    // Initially resting PS: wall adds momentum/work, but no material mass.
    for(int cell=0;cell<grid.size();++cell)state.col(cell)=pureState(1,densities[1]);
    auto pistonRate=evaluate(state,grid,settings);
    const double wallForce=pistonRate.rhs.row(MomentumX).sum()*grid.dx()*grid.dy();
    const double power=pistonRate.rhs.row(TotalEnergy).sum()*grid.dx()*grid.dy();
    near(power-settings.pistonVelocity*state(TotalEnergy,grid.nx-1)*grid.ly,
         settings.pistonVelocity*wallForce);
    // The right translating outflow sweeps in initially resting material.
    near(pistonRate.boundary[AlphaRhoPS],densities[1]*settings.pistonVelocity*grid.ly);
    auto pistonStep=RK3Step(state,grid,settings,1e-8,false,evaluate(state,grid,settings,false));
    validate(pistonStep.state,grid);
    std::cout << "Passed EOS closure, phase-only chemistry, conservative transport, fraction sums, "
                 "piston work and translating-state preservation\n";
}
