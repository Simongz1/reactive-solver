# Kinetics-only 1D hotspot (no mechanics) with an artificially thickened flame.
# Same model as 1D_10elements.i (units: nm, ns), plus a thickening factor F:
#   conduction k -> F k,   rates r1, r2 -> r1 / F, r2 / F
#   => front speed unchanged, front thickness * F
# Two modes:
#   dynamic = 0 : F = Fmax everywhere (pure ATF). The run on mesh dx is then the
#                 exact model on mesh dx / F with the hotspot radius / F and
#                 time * F, so it can be compared with the fine exact runs.
#   dynamic = 1 : F only at the burned/unburned edge (gradient sensor on Y3,
#                 smoothed over omega_l2), F = 1 elsewhere; the hotspot ignites
#                 with the exact kinetics. Optional kernel-size cap (use_cap).
# Front speed: slope of right_front (nm/ns = m/s), or burned_length / 2.

# --- mesh ---------------------------------------------------------------------
nx_el = 10
L_domain = 10000
dx = ${fparse L_domain / nx_el}

# --- thickening ---------------------------------------------------------------
Fmax = 100
dynamic = 1
use_cap = 0
# resolved flame thickness (nm): alpha / s0 = 84.2 nm^2/ns / 76 nm/ns
delta0 = 1.1
# sensor: front_raw = min(1, sensor_gain * dx * |dY3/dx|)
sensor_gain = 3
# smoothing length^2 (nm^2): omega - omega_l2 * lap(omega) = front_raw,
# the mask must reach across the thickened flame (Fmax * delta0), at least 2 dx
omega_l2 = ${fparse max(2 * dx, Fmax * delta0)^2}
# omega peaks at only ~dx / (2 l) on a sharp edge, the gain brings the mask to 1
mask_gain = ${fparse max(4, 2 * sqrt(omega_l2) / dx)}
# kernel-size cap: F <= w / (kernel_c * delta0), w = burned width behind the front
kernel_c = 30
kernel_L = ${fparse max(2 * dx, 2 * kernel_c * Fmax * delta0)}

[Mesh]
        type = GeneratedMesh
        dim = 1
        xmin = 0
        xmax = ${L_domain}
        nx = ${nx_el}
        elem_type = EDGE2
[]

[Variables]
        [T]
        []
        
        #reaction
        [Y1]
                initial_condition = 1
        []

        [Y2]
                initial_condition = 0
        []

        [Y3]
                initial_condition = 0
        []

        # smoothed front sensor and filtered burned fraction (steady filters)
        [omega]
        []
        [kernel_beta]
        []
[]

[AuxVariables]
        [dY3dx]
                order = CONSTANT
                family = MONOMIAL
        []
        [front_raw]
                order = CONSTANT
                family = MONOMIAL
        []
        [Y3_lag]
                order = CONSTANT
                family = MONOMIAL
        []
[]

# sensors are lagged one step so F stays out of the Newton iteration
[AuxKernels]
        [dY3dx]
                type = VariableGradientComponent
                variable = dY3dx
                gradient_variable = Y3
                component = x
                execute_on = 'INITIAL TIMESTEP_BEGIN'
        []
        [front_raw]
                type = ParsedAux
                variable = front_raw
                coupled_variables = 'dY3dx'
                expression = 'min(1, ${sensor_gain} * ${dx} * abs(dY3dx))'
                execute_on = 'INITIAL TIMESTEP_BEGIN'
        []
        [Y3_lag]
                type = ParsedAux
                variable = Y3_lag
                coupled_variables = 'Y3'
                expression = 'Y3'
                execute_on = 'INITIAL TIMESTEP_BEGIN'
        []
[]

[ICs]
        [hotspot]
                type = SmoothCircleIC
                variable = T
                invalue = 1920 # K
                outvalue = 300
                radius = 1000
                x1 = 5000
                y1 = 0
        []
[]

[Materials]
        [activation_energies]
                type = ADGenericConstantMaterial
                prop_names = 'E1 E2'
                prop_values = '24400 23500'
        []
        [pre_exponential_factors]
                type = ADGenericConstantMaterial
                prop_names = 'Z1 Z2'
                prop_values = '32103.943853583012 11810.38092425548'
        []
        [gas_constant]
                type = ADGenericConstantMaterial
                prop_names = 'R'
                prop_values = '1.987'
        []
        [thermal_constants]
                type = ADGenericConstantMaterial
                prop_names = 'rho cv k'
                prop_values = '1820e-9 2355.592 0.361'
        []

        #molecular weight
        [molecular_weight_RDX]
                type = ADGenericConstantMaterial
                prop_names = 'MW'
                prop_values = '0.22226' # kg / mol
        []

        # --- thickening factor ---
        # largest F the burned kernel behind the front can support (Fmax if no cap)
        [thickening_cap]
                type = ADParsedMaterial
                property_name = F_cap
                coupled_variables = 'kernel_beta'
                expression = 'if(${use_cap} > 0, max(1, min(${Fmax}, -${kernel_L} * log(1 - 2 * min(0.49, max(0, kernel_beta))) / (${kernel_c} * ${delta0}))), ${Fmax})'
                outputs = exodus
        []
        [thickening]
                type = ADParsedMaterial
                property_name = F
                coupled_variables = 'omega'
                material_property_names = 'F_cap'
                expression = 'if(${dynamic} > 0, 1 + (F_cap - 1) * min(1, max(0, ${mask_gain} * omega)), ${Fmax})'
                outputs = exodus
        []
        # thickened conductivity F * k
        [k_thick]
                type = ADParsedMaterial
                property_name = k_thick
                material_property_names = 'F k'
                expression = 'F * k'
        []

        #for scaling        
        [k_scaled]
                type = ADParsedMaterial
                material_property_names = 'rho cv k'
                expression = 'k/(rho*cv)'
                property_name = 'k_scaled'
        []
        
        #reaction heat materials
        [Q1]
                type = ADDerivativeParsedMaterial
                derivative_order = 2
                coupled_variables = 'T'
                expression = '-32002.15963286241 + if(T >= 1736.3, 591.0987132187528 * (T-1736.3), 0.0)'
                property_name = 'Q1'
        []
        [Q2]
                type = ADDerivativeParsedMaterial
                derivative_order = 2
                coupled_variables = 'T'
                expression = '7831116.710159273 + if(T >= 1736.3, -722.1193197156482 * (T-1736.3), 0.0)'
                property_name = 'Q2'
        []
        
        #compute reaction rates
        [r1]
                type = ADDerivativeParsedMaterial
                derivative_order = 2
                coupled_variables = 'T'
                material_property_names = 'E1 R Z1 F'
                expression = 'Z1 * exp(-E1/(R*T)) / F'
                property_name = 'r1'
        []
        [r2]
                type = ADDerivativeParsedMaterial
                derivative_order = 2
                coupled_variables = 'T'
                material_property_names = 'E2 R Z2 F'
                expression = 'Z2 * exp(-E2/(R*T)) / F'
                property_name = 'r2'
        []
        
        #second Y2 term
        [r2Y2]
                type = ADDerivativeParsedMaterial
                derivative_order = 2
                coupled_variables = 'T Y2'
                material_property_names = 'r2:=r2(T)'
                expression = 'r2 * Y2'
                property_name = 'r2Y2'
        []
        [r1Y1]
                type = ADDerivativeParsedMaterial
                derivative_order = 2
                coupled_variables = 'T Y1'
                material_property_names = 'r1:=r1(T)'
                expression = 'r1 * Y1'
                property_name = 'r1Y1'
        []

        #decomposition heat
        [QR]
                type = ADDerivativeParsedMaterial
                derivative_order = 2
                coupled_variables = 'T Y1 Y2'
                material_property_names = 'rho cv r1:=r1(T) r2:=r2(T) Q1:=Q1(T) Q2:=Q2(T)'
                expression = 'rho * (Q1 * r1 * Y1 + Q2 * r2 * Y2)'
                property_name = 'QR'
        []
[]

#define residuals

[Kernels]
        #reactions
        [Y1_dot]
                type = ADTimeDerivative
                variable = Y1
        []
        [Y1_source]
                type = ADMatBodyForce
                variable = Y1
                material_property = r1Y1
                value = -1
        []

        #second reaction
        [Y2_dot]
                type = ADTimeDerivative
                variable = Y2
        []
        [Y2_term1]
                type = ADMatBodyForce
                variable = Y2
                material_property = r2Y2
                value = -1
        []
        [Y2_term2]
                type = ADMatBodyForce
                variable = Y2
                material_property = r1Y1
                value = 1
        []
        
        #third reaction
        [Y3_dot]
                type = ADTimeDerivative
                variable = Y3
        []
        [Y3_term1]
                type = ADMatBodyForce
                variable = Y3
                material_property = r2Y2
                value = 1
        []

        #temperature kernels
        [T_dot]
                type = ADHeatConductionTimeDerivative
                variable = T
                density_name = rho
                specific_heat = cv
        []
        [conduction]
                type = ADHeatConduction
                variable = T
                thermal_conductivity = k_thick
        []
        [decomposition]
                type = ADMatHeatSource
                variable = T
                material_property = 'QR'
        []

        # omega - omega_l2 lap(omega) = front_raw
        [omega_self]
                type = Reaction
                variable = omega
        []
        [omega_spread]
                type = MatDiffusion
                variable = omega
                diffusivity = ${omega_l2}
        []
        [omega_source]
                type = CoupledForce
                variable = omega
                v = front_raw
        []
        # kernel_beta - kernel_L^2 lap(kernel_beta) = Y3
        [kernel_beta_self]
                type = Reaction
                variable = kernel_beta
        []
        [kernel_beta_spread]
                type = MatDiffusion
                variable = kernel_beta
                diffusivity = ${fparse kernel_L^2}
        []
        [kernel_beta_source]
                type = CoupledForce
                variable = kernel_beta
                v = Y3_lag
        []
[]

[Postprocessors]
        # burned length (nm); two fronts, so speed per front = slope / 2
        [burned_length]
                type = ElementIntegralVariablePostprocessor
                variable = Y3
                execute_on = 'INITIAL TIMESTEP_END'
        []
        # position of the right front (Y3 = 0.5); its slope is the front speed
        [right_front]
                type = FindValueOnLine
                v = Y3
                target = 0.5
                start_point = '${fparse L_domain / 2} 0 0'
                end_point = '${L_domain} 0 0'
                depth = 30
                error_if_not_found = false
                default_value = ${fparse L_domain / 2}
                execute_on = 'TIMESTEP_END'
        []
        [max_F]
                type = ADElementExtremeMaterialProperty
                mat_prop = F
                value_type = max
                execute_on = 'INITIAL TIMESTEP_END'
        []
[]
[Executioner]
        type = Transient
        solve_type = newton
        petsc_options_iname = '-pc_type -pc_factor_mat_solver_type'
        petsc_options_value = 'lu       mumps'
        nl_rel_tol = 1e-8
        nl_abs_tol = 1e-10
        nl_max_its = 15
        
        [TimeStepper]
                type = FunctionDT
                function = timestep
                min_dt = 1e-8
        []
        
        l_tol = 1e-8
        automatic_scaling = true
        end_time = 10000
        line_search = 'bt'
        
        [Quadrature]
                type = TRAP
                order = FIRST
        []
[]

[Functions]
        [./timestep]
                type = ParsedFunction
                expression = '1e-1'
        [../]
[]

[Preconditioning]
        [smp]
                type = SMP
                full = true
        []
[]

[VectorPostprocessors]
        [profiles]
                type = LineValueSampler
                variable = 'T Y1 Y2 Y3 omega kernel_beta'
                start_point = '0 0 0'
                end_point = '10000 0 0'
                # 250 elements have 251 nodes, including both endpoints.
                num_points = 251
                sort_by = x
                execute_on = 'INITIAL TIMESTEP_END'
        []
[]

[Outputs]
        exodus = true
        csv = true
        time_step_interval = 5
[]      
