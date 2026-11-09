# Reactive Solver

This repository contains the required code to compile and run 1D and 2D finite difference schemes for coupled thermo-chemical and thermomechanical-chemical models of thermal decomposition and deflagration of energetic materials.

The spatial discretization uses a second-order finite difference stencil, and a Strong Stability-Preserving Runge-Kutta of order 3 (SSP RK3).

## Kinetics
Local decomposition follows the coupled system
$$\dot Y_1=-r_1Y_1$$
$$\dot Y_2=r_1Y_1-r_2Y_2$$
$$\dot Y_3=r_2Y_2$$

With the closure condition $Y_1+Y_2+Y_3=1$ implied.

Each reaction rate follows an Arrhenius function of the form

$$r_i=Z_ie^{-E_i/RT}$$

where $Z_i$ is a pre-exponential rate factor for reaction $i$ in units of inverse time, $E_i$ is the activation energy for reaction $i$, $R$ is the ideal gas constant, and $T$ is temperature.

## Thermal Transport
The evolution of the temperature field is governed by the equation

$$\rho c_v\frac{\partial T}{\partial t}=\nabla\cdot k\nabla T+\rho(r_1Y_1Q_1+r_2Y_2Q_2) + T\frac{\partial\boldsymbol\sigma}{\partial T}:\text{tr}\mathbf{D}$$

where $\rho$ and #c_v# are the density and heat capacity, $k$ is the thermal conductivity, $Q_1$ and $Q_2$ are the heats of reaction of reaction 1 and 2, respectively, $\boldsymbol\sigma$ is the Cauchy stress in the spatial configuration, and $\mathbf{D}$ is the rate of deformation tensor. For a local form, and for a stress tensor whose temperature dependence comes only from the pressure, this term becomes

$$\text{tr}\mathbf{D}=\frac{\dot J}{J}$$

where $J=\det\mathbf{F}=\rho_0/\rho$ is the relative change in volume.

## Stress Model

The present model makes the simplification that 

$$\boldsymbol{\sigma}=-p\mathbf{I}$$

where the term $p$ is a thermodynamic pressure computed from an equation of state (EOS) of the form

$$p_i=A_ie^{-R_{1,i} J} + B_ie^{-R_{2,i}J} + \omega\rho c_v (T-T_0)$$

where the terms $A_i$ and $B_i$ posses units of pressure, $R_{1,i}$ and $R_{2,i}$ are dimensionless parameters, $\omega$ is the Grüneisen coefficient, and $T_0=300\text{K}$ is the temperature at the reference state.

## Balance of Mass and Linear Momentum

The conservative form of the evolution equations for density and momentum take the form

$$\frac{\partial\rho}{\partial t}+\nabla\cdot(\rho\mathbf{v})=0$$
$$\frac{\partial\mathbf{m}}{\partial t}+\nabla\cdot\left(\frac{\mathbf{m}\otimes\mathbf{m}}{\rho}+p\mathbf{I}\right)=0$$

where $\mathbf{m}$ is the momentum vector in cartesian coordinates.