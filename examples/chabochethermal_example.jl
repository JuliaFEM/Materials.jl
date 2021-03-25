# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Printf
using Tensors
using Plots
using Test
using DelimitedFiles
using Materials
pyplot()

let
    # https://rosettacode.org/wiki/Align_columns#Julia
    # left/right/center justification of strings:
    ljust(s::String, width::Integer) = s * " "^max(0, width - length(s))
    # rjust(s::String, width::Integer) = " "^max(0, width - length(s)) * s
    # function center(s::String, width::Integer)
    #     pad = width - length(s)
    #     if pad <= 0
    #         return s
    #     else
    #         pad2 = div(pad, 2)
    #         return " "^pad2 * s * " "^(pad - pad2)
    #     end
    # end

    """    format_numbers(xx::Array{<:Real})

    Format a rank-1 array of numbers to "%0.6g", align the ones column, and pad to the same length.

    Return a rank-1 array of the resulting strings.
    """
    function format_numbers(xx::Array{<:Real})  # TODO: extend to handle complex numbers, too
        # - real numbers x for which |x| < 1 always have "0." at the start
        # - e-notation always has a dot
        function find_ones_column(s::String)
            dot_column = findfirst(".", s)
            ones_column = (dot_column !== nothing) ? (dot_column[1] - 1) : length(s)
            @assert (ones_column isa Integer) "failed to detect column for ones"
            return ones_column
        end

        ss = [@sprintf("%0.6g", x) for x in xx]

        ones_columns = [find_ones_column(s) for s in ss]
        ones_target_column = maximum(ones_columns)
        left_pads = ones_target_column .- ones_columns
        @assert all(p >= 0 for p in left_pads) "negative padding length"
        ss = [" "^p * s for (s, p) in zip(ss, left_pads)]

        max_length = maximum(length(s) for s in ss)
        ss = [ljust(s, max_length) for s in ss]

        return ss
    end

    function constant(value::Real)
        function interpolate(x::Real)
            # `x` may be a `ForwardDiff.Dual` even when `value` is a float.
            return convert(typeof(x), value)
        end
        return interpolate
    end

    function capped_linear(x1::Real, y1::Real, x2::Real, y2::Real)
        if x1 > x2
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        end
        dx = x2 - x1
        dx > 0 || error("must have x2 > x1")
        dy = y2 - y1
        function interpolate(x::Real)
            alpha = (x - x1) / dx
            alpha = max(0, min(alpha, 1))
            return y1 + alpha * dy
        end
        return interpolate
    end

    """Celsius to Kelvin."""
    function K(degreesC::Real)
        return degreesC + 273.15
    end

    """Kelvin to Celsius."""
    function degreesC(K::Real)
        return K - 273.15
    end

    let T0 = K(20.0),
        T1 = K(620.0),
        # Thermal elongation, Eurocode, SFS-EN 1993-1-2, carbon steel
        #   1.2e-5 * T[C°] + 0.4e-8 * T[C°]^2 - 2.416e-4
        # α is the derivative of this.
        #
        #   using SymPy
        #   @vars T real=true
        #   thermal_elongation = 1.2e-5 * T + 0.4e-8 * T^2 - 2.416e-4
        #   alpha = diff(thermal_elongation, T)
        #   alpha0 = subs(alpha, (T, 20))   # 1.216e-5
        #   alpha1 = subs(alpha, (T, 600))  # 1.680e-5
        #
        # See also:
        #     https://www.engineeringtoolbox.com/linear-expansion-coefficients-d_95.html
        parameters = ChabocheThermalParameterState(theta0=T0,
                                                   E=capped_linear(T0, 200.0e3, T1, 100.0e3),
                                                   #nu=capped_linear(T0, 0.3, T1, 0.35),
                                                   nu=constant(0.3),
                                                   #alpha=capped_linear(T0, 1.216e-5, T1, 1.680e-5),
                                                   alpha=constant(1.216e-5),
                                                   R0=capped_linear(T0, 100.0, T1, 50.0),
                                                   # R0=constant(1000.0),
                                                   # viscous hardening in constant strain rate test: (tvp * ε')^(1/nn) * Kn
                                                   tvp=1000.0,
                                                   Kn=capped_linear(T0, 100.0, T1, 50.0),
                                                   nn=capped_linear(T0, 1.0, T1, 4.0),
                                                   # C1=constant(10000.0),
                                                   # D1=constant(100.0),
                                                   # C2=constant(50000.0),
                                                   # D2=constant(1000.0),
                                                   C1=constant(1000.0),
                                                   D1=constant(10.0),
                                                   C2=constant(0.0),
                                                   D2=constant(0.0),
                                                   C3=constant(0.0),
                                                   D3=constant(0.0),
                                                   # Q=capped_linear(T0, 50.0, T1, 10.0),
                                                   # b=capped_linear(T0, 100.0, T1, 0.01)),
                                                   Q=constant(0.0),
                                                   b=constant(0.0)),
        # uniaxial pull test, so we set only dε11.
        # stress_rate=10.0, # dσ/dt [MPa/s] (for stress-driven test)
        strain_rate=1e-3,  # dε/dt [1/s] (for strain-driven test)
        strain_final=0.005,  # when to stop the pull test
        dt=0.05,  # simulation timestep, [s]
        # dstress11 = stress_rate * dt,  # dσ11 during one timestep (stress-driven)
        dstrain11 = strain_rate * dt,  # dε11 during one timestep (strain-driven)
        n_timesteps = Integer(round(strain_final / dstrain11)),
        #constant_temperatures = range(T0, T1, length=3),
        constant_temperatures = [K(20.0), K(150.0), K(300.0), K(620.0)],
        timevar_temperature = range(T0, T0 + 130, length=n_timesteps + 1)

        # TODO: Improve the plotting to use two separate figures so that we can plot these examples too
        # TODO: (without making individual plots too small).
        # TODO: Plots.jl can't currently do that; investigate whether the underlying PyPlot.jl can.

        # p1 = plot()  # make empty figure
        #
        #
        # # --------------------------------------------------------------------------------
        # # constant temperature, constant strain rate pull test
        #
        # println("Constant temperature tests")
        # for T in constant_temperatures
        #     println("T = $(degreesC(T))°C")
        #     mat = ChabocheThermal(parameters=parameters)
        #     mat.drivers.temperature = T
        #     mat.ddrivers.temperature = 0
        #     stresses = [mat.variables.stress[1,1]]
        #     strains = [mat.drivers.strain[1,1]]
        #     for i in 1:n_timesteps
        #         uniaxial_increment!(mat, dstrain11, dt)
        #         # stress_driven_uniaxial_increment!(mat, dstress11, dt)
        #         update_material!(mat)
        #         push!(strains, mat.drivers.strain[1,1])
        #         push!(stresses, mat.variables.stress[1,1])
        #     end
        #     println("    ε11, σ11, at end of simulation")
        #     println("    $(strains[end]), $(stresses[end])")
        #     plot!(strains, stresses, label="\$\\sigma(\\varepsilon)\$ @ \$$(degreesC(T))°C\$")
        # end
        #
        #
        # # --------------------------------------------------------------------------------
        # # varying temperature, constant strain rate pull test
        #
        # println("Time-varying temperature tests (activates ΔT terms)")
        # println("T = $(degreesC(timevar_temperature[1]))°C ... $(degreesC(timevar_temperature[end]))°C, linear profile.")
        # mat = ChabocheThermal(parameters=parameters)
        # stresses = [mat.variables.stress[1,1]]
        # strains = [mat.drivers.strain[1,1]]
        # temperature_pairs = zip(timevar_temperature, timevar_temperature[2:end])
        # for (Tcurr, Tnext) in temperature_pairs
        #     # println("        Tcurr = $(degreesC(Tcurr))°C, Tnext = $(degreesC(Tnext))°C, ΔT = $(Tnext - Tcurr)°C")
        #     mat.drivers.temperature = Tcurr
        #     mat.ddrivers.temperature = Tnext - Tcurr
        #     uniaxial_increment!(mat, dstrain11, dt)
        #     # stress_driven_uniaxial_increment!(mat, dstress11, dt)
        #     update_material!(mat)
        #     push!(strains, mat.drivers.strain[1,1])
        #     push!(stresses, mat.variables.stress[1,1])
        # end
        # println("    ε11, σ11, at end of simulation")
        # println("    $(strains[end]), $(stresses[end])")
        # plot!(strains, stresses, label="\$\\sigma(\\varepsilon)\$ @ $(degreesC(timevar_temperature[1]))°C ... $(degreesC(timevar_temperature[end]))°C")
        #
        # xlabel!("\$\\varepsilon\$")
        # ylabel!("\$\\sigma\$")
        # title!("Uniaxial pull test (strain-driven)")
        #
        #
        # # --------------------------------------------------------------------------------
        # # cyclic temperature/strain
        # #
        # #  - boomerang/fan in elastic region, no hysteresis
        # #  - check that the endpoint stays the same
        # #    - It doesn't when temperature effects are enabled; linearly dt-dependent drift; from the integrator?
        #
        # println("Elastic behavior under cyclic loading")
        # function halfcycle(x0, x1, n)
        #     return x0 .+ (x1 - x0) .* range(0, 1, length=n)
        # end
        # function cycle(x0, x1, halfn)  # 2 * halfn - 1 steps in total (duplicate at middle omitted)
        #     return cat(halfcycle(x0, x1, halfn),
        #                halfcycle(x1, x0, halfn)[2:end],
        #                dims=1)
        # end
        #
        # strain_rate = 1e-4  # uniaxial constant strain rate, [1/s]
        # cycle_time = 10.0  # one complete cycle, [s]
        # ncycles = 20
        # n = 201  # points per half-cycle (including endpoints; so n - 1 timesteps per half-cycle)
        #
        # Ta = T0  # temperature at cycle start, [K]
        # Tb = K(50.0)  # temperature at maximum strain (at cycle halfway point), [K]
        #
        # # Observe that:
        # strain_max = strain_rate * (cycle_time / 2)
        # dt = cycle_time / (2 * (n - 1))
        #
        # description = "$(ncycles) cycles, εₘₐₓ = $(strain_max), Ta = $(degreesC(Ta))°C, Tb = $(degreesC(Tb))°C"
        # println("    $(description)")
        # mat = ChabocheThermal(parameters=parameters)
        # stresses = [mat.variables.stress[1,1]]
        # strains = [mat.drivers.strain[1,1]]
        # temperatures = cycle(Ta, Tb, n)
        # temperature_pairs = zip(temperatures, temperatures[2:end])
        # dstrain11 = strain_rate * dt  # = strain_rate * (cycle_time / 2) / (n - 1) = strain_max / (n - 1)
        # dstrain11s = cat(repeat([dstrain11], n - 1),
        #                  repeat([-dstrain11], n - 1),
        #                  dims=1)
        # for cycle in 1:ncycles
        #     cycle_str = @sprintf("%02d", cycle)
        #     println("    start cycle $(cycle_str), ε11 = $(strains[end]), σ11 = $(stresses[end])")
        #     for ((Tcurr, Tnext), dstrain) in zip(temperature_pairs, dstrain11s)
        #         mat.drivers.temperature = Tcurr
        #         mat.ddrivers.temperature = Tnext - Tcurr
        #         uniaxial_increment!(mat, dstrain, dt)
        #         # stress_driven_uniaxial_increment!(mat, dstress11, dt)
        #         update_material!(mat)
        #         push!(strains, mat.drivers.strain[1,1])
        #         push!(stresses, mat.variables.stress[1,1])
        #     end
        # end
        # println("    ε11, σ11, at end of simulation")
        # println("    $(strains[end]), $(stresses[end])")
        # # println("    $(mat.variables.plastic_strain[end])")
        # p2 = plot(strains, stresses, label="\$\\sigma(\\varepsilon)\$")
        #
        # # plot!(xx2, yy2, label="...")  # to add new curves into the same figure
        # xlabel!("\$\\varepsilon\$")
        # ylabel!("\$\\sigma\$")
        # title!("Elastic test, $(description)")
        #
        #
        # # --------------------------------------------------------------------------------
        # # non-symmetric cyclic loading
        # #
        # # Strain-driven case. Should exhibit stress relaxation.
        #
        # println("Non-symmetric strain cycle")
        # strain_rate = 1e-3  # uniaxial constant strain rate, [1/s]
        # cycle_time = 5.0  # one complete cycle, [s]
        # ncycles = 20
        # n = 51  # points per half-cycle (including endpoints; so n - 1 timesteps per half-cycle)
        #
        # Ta = T0  # temperature at simulation start, [K]
        # Tb = K(50.0)  # temperature at maximum strain (at cycle halfway point), [K]
        # Tm = Ta + (Tb - Ta) / 2  # temperature at start of each cycle, [K]
        #
        # strain_max = strain_rate * cycle_time  # accounting for initial loading, too.
        # dt = cycle_time / (2 * (n - 1))
        #
        # description = "$(ncycles) cycles, εₘₐₓ = $(strain_max), Ta = $(degreesC(Ta))°C, Tb = $(degreesC(Tb))°C"
        # println("    $(description)")
        # mat = ChabocheThermal(parameters=parameters)  # TODO: always use the AF model here (one backstress).
        # stresses = [mat.variables.stress[1,1]]
        # strains = [mat.drivers.strain[1,1]]
        #
        # # initial loading
        # temperatures = halfcycle(Ta, Tm, n)
        # temperature_pairs = zip(temperatures, temperatures[2:end])
        # dstrain11 = strain_rate * dt
        # dstrain11s = repeat([dstrain11], n - 1)
        #
        # for ((Tcurr, Tnext), dstrain) in zip(temperature_pairs, dstrain11s)
        #     mat.drivers.temperature = Tcurr
        #     mat.ddrivers.temperature = Tnext - Tcurr
        #     uniaxial_increment!(mat, dstrain, dt)
        #     # stress_driven_uniaxial_increment!(mat, dstress11, dt)
        #     update_material!(mat)
        #     push!(strains, mat.drivers.strain[1,1])
        #     push!(stresses, mat.variables.stress[1,1])
        # end
        #
        # # cycles
        # eps0 = strains[end]  # for marking the start of the first cycle in the figure
        # sig0 = stresses[end]
        # temperatures = cycle(Tm, Tb, n)
        # temperature_pairs = zip(temperatures, temperatures[2:end])
        # dstrain11 = strain_rate * dt
        # dstrain11s = cat(repeat([dstrain11], n - 1),
        #                  repeat([-dstrain11], n - 1),
        #                  dims=1)
        # cycle_midpoint = n - 1
        #
        # for cycle in 1:ncycles
        #     cycle_str = @sprintf("%02d", cycle)
        #     println("    cycle $(cycle_str)")
        #     data_to_print = []
        #     for (k, ((Tcurr, Tnext), dstrain)) in enumerate(zip(temperature_pairs, dstrain11s))
        #         if k == 1 || k == cycle_midpoint
        #             push!(data_to_print, (strains[end], stresses[end]))
        #         end
        #
        #         mat.drivers.temperature = Tcurr
        #         mat.ddrivers.temperature = Tnext - Tcurr
        #         uniaxial_increment!(mat, dstrain, dt)
        #         # stress_driven_uniaxial_increment!(mat, dstress11, dt)
        #         update_material!(mat)
        #         push!(strains, mat.drivers.strain[1,1])
        #         push!(stresses, mat.variables.stress[1,1])
        #     end
        #
        #     strains_to_print, stresses_to_print = (collect(col) for col in zip(data_to_print...))
        #     strains_to_print = format_numbers(strains_to_print)
        #     stresses_to_print = format_numbers(stresses_to_print)
        #     println("        start    ε11 = $(strains_to_print[1]), σ11 = $(stresses_to_print[1])")
        #     println("        midpoint ε11 = $(strains_to_print[2]), σ11 = $(stresses_to_print[2])")
        # end
        #
        # p3 = plot(strains, stresses, label="\$\\sigma(\\varepsilon)\$")
        # scatter!([eps0], [sig0], markercolor=:blue, label="First cycle start")
        # xlabel!("\$\\varepsilon\$")
        # ylabel!("\$\\sigma\$")
        # title!("Non-symmetric strain cycle, $(ncycles) cycles")
        #
        #
        # # --------------------------------------------------------------------------------
        # # stress-driven non-symmetric cycle
        # #
        # #   - AF (Chaboche with one kinematic hardening backstress) should lead to constant
        # #     ratcheting strain per stress cycle.
        #
        # println("Non-symmetric stress cycle")
        # stress_rate = 40.0  # uniaxial constant stress rate, [MPa/s]
        # cycle_time = 5.0  # one complete cycle, [s]
        # ncycles = 40
        # n = 51  # points per half-cycle (including endpoints; so n - 1 timesteps per half-cycle)
        #
        # Ta = T0  # temperature at simulation start, [K]
        # Tb = K(50.0)  # temperature at maximum strain (at cycle halfway point), [K]
        # Tm = Ta + (Tb - Ta) / 2  # temperature at start of each cycle, [K]
        #
        # strain_max = strain_rate * cycle_time  # accounting for initial loading, too.
        # dt = cycle_time / (2 * (n - 1))
        #
        # description = "$(ncycles) cycles, εₘₐₓ = $(strain_max), Ta = $(degreesC(Ta))°C, Tb = $(degreesC(Tb))°C"
        # println("    $(description)")
        # mat = ChabocheThermal(parameters=parameters)  # TODO: always use the AF model here (one backstress).
        # stresses = [mat.variables.stress[1,1]]
        # strains = [mat.drivers.strain[1,1]]
        #
        # # initial loading
        # temperatures = halfcycle(Ta, Tm, n)
        # temperature_pairs = zip(temperatures, temperatures[2:end])
        # dstress11 = stress_rate * dt
        # dstress11s = repeat([dstress11], n - 1)
        #
        # for ((Tcurr, Tnext), dstress) in zip(temperature_pairs, dstress11s)
        #     mat.drivers.temperature = Tcurr
        #     mat.ddrivers.temperature = Tnext - Tcurr
        #     stress_driven_uniaxial_increment!(mat, dstress, dt)
        #     update_material!(mat)
        #     push!(strains, mat.drivers.strain[1,1])
        #     push!(stresses, mat.variables.stress[1,1])
        # end
        #
        # # cycles
        # eps0 = strains[end]
        # sig0 = stresses[end]
        # temperatures = cycle(Tm, Tb, n)
        # temperature_pairs = zip(temperatures, temperatures[2:end])
        # dstress11 = stress_rate * dt
        # dstress11s = cat(repeat([dstress11], n - 1),
        #                  repeat([-dstress11], n - 1),
        #                  dims=1)
        # cycle_midpoint = n - 1
        #
        # cycle_start_strains = convert(Array{Float64}, [])  # TODO: what's the julianic way to do this?
        # for cycle in 1:ncycles
        #     cycle_str = @sprintf("%02d", cycle)
        #     println("    cycle $(cycle_str)")
        #     push!(cycle_start_strains, strains[end])
        #     data_to_print = []
        #     for (k, ((Tcurr, Tnext), dstress)) in enumerate(zip(temperature_pairs, dstress11s))
        #         if k == 1 || k == cycle_midpoint
        #             push!(data_to_print, (strains[end], stresses[end]))
        #         end
        #
        #         mat.drivers.temperature = Tcurr
        #         mat.ddrivers.temperature = Tnext - Tcurr
        #         stress_driven_uniaxial_increment!(mat, dstress, dt)
        #         update_material!(mat)
        #         push!(strains, mat.drivers.strain[1,1])
        #         push!(stresses, mat.variables.stress[1,1])
        #     end
        #
        #     strains_to_print, stresses_to_print = (collect(col) for col in zip(data_to_print...))
        #     strains_to_print = format_numbers(strains_to_print)
        #     stresses_to_print = format_numbers(stresses_to_print)
        #     println("        start    ε11 = $(strains_to_print[1]), σ11 = $(stresses_to_print[1])")
        #     println("        midpoint ε11 = $(strains_to_print[2]), σ11 = $(stresses_to_print[2])")
        # end
        #
        # println("Strain at cycle start:")
        # cycle_start_strains_to_print = format_numbers(cycle_start_strains)
        # diffs = diff(cycle_start_strains)
        # diffs_to_print = cat([nothing], format_numbers(diffs), dims=1)
        # for (cycle, (strain, dstrain)) in enumerate(zip(cycle_start_strains_to_print, diffs))
        #     cycle_str = @sprintf("%02d", cycle)
        #     println("    cycle $(cycle_str), ε11 = $(strain), Δε11 w.r.t. previous cycle = $(dstrain)")
        # end
        #
        # p4 = plot(strains, stresses, label="\$\\sigma(\\varepsilon)\$")
        # scatter!([eps0], [sig0], markercolor=:blue, label="First cycle start")
        # xlabel!("\$\\varepsilon\$")
        # ylabel!("\$\\sigma\$")
        # title!("Non-symmetric stress cycle, $(ncycles) cycles")
        #
        #
        # # --------------------------------------------------------------------------------
        # # TODO:
        # # - more tests based on Bari's thesis
        # # - we need to implement pure plasticity (compute dotp from the consistency condition)
        # #   in order to compare to Bari's results.
        # #
        # # 1 ksi = 6.8947572932 MPa
        # #
        # # From Bari's thesis, paper 1, p. 25 (PDF page 31):
        # #
        # # E = 26300 ksi = 181332.11681116 MPa
        # # ν = 0.302
        # # σ₀ = 18.8 ksi = 129.62143711216 MPa (initial yield)
        # # C₁ = 60000 ksi = 413685.437592 MPa
        # # C₂ = 12856 ksi = 88638.9997613792 MPa
        # # C₃ = 455 ksi = 3137.1145684059998 MPa
        # # γ₁ = 20000 (D₁ in Materials.jl)
        # # γ₂ = 800
        # # γ₃ = 9
        # #
        # # From the article text and figure captions, these values seem to be for CS1026 steel.
        # #
        # # c = 6.8947572932  # MPa/ksi
        # # parameters = ChabocheThermalParameterState(theta0=T0,
        # #                                            E=constant(26300*c),
        # #                                            nu=constant(0.302),
        # #                                            alpha=constant(1.216e-5),  # not used in tests based on Bari
        # #                                            R0=constant(18.8*c),
        # #                                            # viscous hardening in constant strain rate test: (tvp * ε')^(1/nn) * Kn
        # #                                            tvp=1000.0,
        # #                                            Kn=constant(0.0),  # TODO
        # #                                            nn=constant(0.0),  # TODO
        # #                                            C1=constant(60000*c),
        # #                                            D1=constant(20000),
        # #                                            C2=constant(12856*c),
        # #                                            D2=constant(800),
        # #                                            C3=constant(455*c),
        # #                                            D3=constant(9),
        # #                                            Q=constant(0.0),
        # #                                            b=constant(0.0))
        #
        # # --------------------------------------------------------------------------------
        # # plot the results
        #
        # # https://docs.juliaplots.org/latest/layouts/
        # plot(p1, p2, p3, p4, layout=(2, 2))

        # Abaqus as reference point. Data provided by Joona.
        # The data describes a strain-driven uniaxial cyclic push-pull test in the 22 direction.
        let path = joinpath("test_chabochethermal", "chabochethermal_cyclic_test_no_autostep.rpt"),
            data = readdlm(path, Float64; skipstart=4),
            ts = data[:, 1],
            e11_ = data[:, 2],  # note Abaqus component ordering
            e12_ = data[:, 3],
            e13_ = data[:, 4],
            e22_ = data[:, 5],
            e23_ = data[:, 6],
            e33_ = data[:, 7],
            s11_ = data[:, 8],
            s12_ = data[:, 9],
            s13_ = data[:, 10],
            s22_ = data[:, 11],
            s23_ = data[:, 12],
            s33_ = data[:, 13],
            cumeq_ = data[:, 14],
            temperature_ = data[:, 15],
            # note our component ordering (standard Voigt)
            strains = [[e11_[i], e22_[i], e33_[i], e23_[i], e13_[i], e12_[i]] for i in 1:length(ts)],
            stresses = [[s11_[i], s22_[i], s33_[i], s23_[i], s13_[i], s12_[i]] for i in 1:length(ts)],
            T0 = K(23.0),
            T1 = K(400.0),
            parameters = ChabocheThermalParameterState(theta0=T0,
                                                       E=capped_linear(T0, 200.0e3, T1, 120.0e3),
                                                       nu=capped_linear(T0, 0.3, T1, 0.45),
                                                       alpha=capped_linear(K(0.0), 1.0e-5, T1, 1.5e-5),
                                                       R0=capped_linear(T0, 100.0, T1, 50.0),
                                                       # viscous hardening in constant strain rate test: (tvp * ε')^(1/nn) * Kn
                                                       tvp=1.0,
                                                       Kn=capped_linear(T0, 50.0, T1, 250.0),
                                                       nn=capped_linear(T0, 10.0, T1, 3.0),
                                                       C1=capped_linear(T0, 100000.0, T1, 20000.0),
                                                       D1=constant(1000.0),
                                                       C2=capped_linear(T0, 10000.0, T1, 2000.0),
                                                       D2=constant(100.0),
                                                       C3=capped_linear(T0, 1000.0, T1, 200.0),
                                                       D3=constant(10.0),
                                                       Q=capped_linear(T0, 100.0, T1, 50.0),
                                                       b=capped_linear(T0, 50.0, T1, 10.0)),
            mat = ChabocheThermal(parameters=parameters)

            time_pairs = zip(ts, ts[2:end])
            strain_pairs = zip(strains, strains[2:end])
            stress_pairs = zip(stresses, stresses[2:end])

            thetas = [K(celsius) for celsius in temperature_]
            temperature_pairs = zip(thetas, thetas[2:end])

            ts_output = [copy(mat.drivers.time)]
            es = [copy(mat.drivers.strain)]
            ss = [copy(mat.variables.stress)]
            X1s = [copy(mat.variables.X1)]
            X2s = [copy(mat.variables.X2)]
            X3s = [copy(mat.variables.X3)]
            Rs = [copy(mat.variables.R)]
            flags = [false]  # plastic response activation flag (computed from output)
            for (step, ((tcurr_, tnext_), (Tcurr_, Tnext_), (ecurr_, enext_), (scurr_, snext_))) in enumerate(zip(time_pairs,
                                                                                                temperature_pairs,
                                                                                                strain_pairs,
                                                                                                                 stress_pairs))
                print("$(step) out of $(length(time_pairs)), t = $(tcurr_)...\n")
                cumeq_old = mat.variables.cumeq  # for plastic response activation detection

                dtime_ = tnext_ - tcurr_
                dtemperature_ = Tnext_ - Tcurr_
                dstrain_ = enext_ - ecurr_
                dstress_ = snext_ - scurr_

                if dtime_ < 1e-8
                    print("    zero Δt in input data, skipping\n")
                    continue
                end

                # Use a smaller timestep internally and gather results every N timesteps.
                # We have just backward Euler for now, so the integrator is not very accurate.
                # TODO: We offer this substepping possibility to obtain higher accuracy.
                # TODO: We don't know what Abaqus internally does here when autostep is on.
                # TODO: Likely, it uses some kind of error indicator and adapts the
                # TODO: timestep size based on that.
                # TODO: Should disable autostep and recompute the Abaqus reference results, so that we can be sure of what the results mean.
                N = 1
                for substep in 1:N
                    tcurr = tcurr_ + ((substep - 1) / N) * dtime_
                    tnext = tcurr_ + (substep / N) * dtime_
                    Tcurr = Tcurr_ + ((substep - 1) / N) * dtemperature_
                    Tnext = Tcurr_ + (substep / N) * dtemperature_
                    ecurr = ecurr_ + ((substep - 1) / N) * dstrain_
                    enext = ecurr_ + (substep / N) * dstrain_
                    scurr = scurr_ + ((substep - 1) / N) * dstress_
                    snext = scurr_ + (substep / N) * dstress_
                    dtime = tnext - tcurr
                    dtemperature = Tnext - Tcurr

                    mat.drivers.temperature = Tcurr  # value at start of timestep
                    mat.ddrivers.time = dtime
                    mat.ddrivers.temperature = dtemperature

                    # # For reference only:
                    # # This is how we would use the whole strain tensor from the Abaqus data as driver.
                    # #
                    # dstrain = enext - ecurr
                    # mat.ddrivers.strain = fromvoigt(Symm2{Float64}, dstrain, offdiagscale=2.0)
                    # integrate_material!(mat)

                    # This one is the actual test setup.
                    # Strain-driven uniaxial pull test in 22 direction, using only ε22 data as driver.
                    #
                    # note: our component ordering (Julia's standard Voigt)
                    dstrain22 = (enext - ecurr)[2]
                    dstrain_knowns = [missing, dstrain22, missing, missing, missing, missing]
                    dstrain_initialguess = [-dstrain22 * mat.parameters.nu(Tcurr),
                                            dstrain22,
                                            -dstrain22 * mat.parameters.nu(Tcurr),
                                            0.0, 0.0, 0.0]
                    general_increment!(mat, dstrain_knowns, dt, dstrain_initialguess)

                    # # For reference only:
                    # # This is how we would do this for a stress-driven test, using the σ22 data as driver.
                    # #
                    # # note: our component ordering (Julia's standard Voigt)
                    # dstress22 = (snext - scurr)[2]
                    # dstress_knowns = [missing, dstress22, missing, missing, missing, missing]
                    # dstrain22_initialguess = dstress22 / mat.parameters.E(Tcurr)
                    # dstrain_initialguess = [-dstrain22_initialguess * mat.parameters.nu(Tcurr),
                    #                         dstrain22_initialguess,
                    #                         -dstrain22_initialguess * mat.parameters.nu(Tcurr),
                    #                         0.0, 0.0, 0.0]
                    # stress_driven_general_increment!(mat, dstress_knowns, dt, dstrain_initialguess)

                    update_material!(mat)
                end

                push!(ts_output, tnext_)
                push!(es, copy(mat.drivers.strain))
                push!(ss, copy(mat.variables.stress))
                push!(X1s, copy(mat.variables.X1))
                push!(X2s, copy(mat.variables.X2))
                push!(X3s, copy(mat.variables.X3))
                push!(Rs, copy(mat.variables.R))

                plastic_active = (mat.variables.cumeq != cumeq_old)
                push!(flags, plastic_active)
            end
            # print("reference\n")
            # print(e33_)
            # print("\nresult\n")
            # print(e33s)
#            @test isapprox(e33s, e33_; rtol=0.05)

            # ------------------------------------------------------------
            current_run = Int64[]
            runs = Array{typeof(current_run), 1}()
            if flags[1]  # signal may be "on" at start
                push!(current_run, 1)
            end
            for (k, (flag1, flag2)) in enumerate(zip(flags, flags[2:end]))
                if flag2 && !flag1  # signal switches on in this interval
                    @assert length(current_run) == 0
                    push!(current_run, k + 1)  # run starts at the edge where the signal is "on"
                elseif flag1 && !flag2  # signal switches off in this interval
                    @assert length(current_run) == 1
                    push!(current_run, k)  # run ends at the edge where the signal was last "on"
                    push!(runs, current_run)
                    current_run = Int64[]
                end
            end
            if flags[end]  # signal may be "on" at end
                push!(current_run, length(flags))
                push!(runs, current_run)
                current_run = Int64[]
            end
            @assert length(current_run) == 0
            # ------------------------------------------------------------

            e11s = [strain[1,1] for strain in es]
            e22s = [strain[2,2] for strain in es]
            s11s = [stress[1,1] for stress in ss]
            s22s = [stress[2,2] for stress in ss]
            X1_11s = [X1[1,1] for X1 in X1s]
            X1_22s = [X1[2,2] for X1 in X1s]
            X2_11s = [X2[1,1] for X2 in X2s]
            X2_22s = [X2[2,2] for X2 in X2s]
            X3_11s = [X3[1,1] for X3 in X3s]
            X3_22s = [X3[2,2] for X3 in X3s]

            # debug
            p1 = plot()
            plot!(ts, e22_, label="\$\\varepsilon_{22}\$ (Abaqus)")
            plot!(ts, e11_, label="\$\\varepsilon_{11}\$ (Abaqus)")
            plot!(ts_output, e22s, label="\$\\varepsilon_{22}\$ (Materials.jl)")
            plot!(ts_output, e11s, label="\$\\varepsilon_{11}\$ (Materials.jl)")
            for (s, e) in runs
                plot!(ts_output[s:e], e22s[s:e], linecolor=:black, label=nothing)
            end
            plot!([NaN], [NaN], linecolor=:black, label="plastic response active")

            p2 = plot()
            plot!(ts, s22_, label="\$\\sigma_{22}\$ [MPa] (Abaqus)")
            plot!(ts, s11_, label="\$\\sigma_{11}\$ [MPa] (Abaqus)")
            plot!(ts_output, s22s, label="\$\\sigma_{22}\$ [MPa] (Materials.jl)")
            plot!(ts_output, s11s, label="\$\\sigma_{11}\$ [MPa] (Materials.jl)")
            # scatter!(ts[flags], s22s[flags], markersize=3, markercolor=:black, markershape=:rect, label="in plastic region")
            for (s, e) in runs
                plot!(ts_output[s:e], s22s[s:e], linecolor=:black, label=nothing)
            end
            plot!([NaN], [NaN], linecolor=:black, label="plastic response active")

            p3 = plot()
            plot!(ts, temperature_, label="\$\\theta\$ [°C]")
            plot!(ts_output, Rs, label="\$R\$ [MPa]")  # stress-like, unrelated, but the range of values fits here best.
            for (s, e) in runs
                plot!(ts[s:e], temperature_[s:e], linecolor=:black, label=nothing)
            end
            plot!([NaN], [NaN], linecolor=:black, label="plastic response active")

            p4 = plot()
            plot!(ts_output, X1_22s, label="\$(X_1)_{22}\$ [MPa]")
            plot!(ts_output, X1_11s, label="\$(X_1)_{11}\$ [MPa]")
            for (s, e) in runs
                plot!(ts_output[s:e], X1_22s[s:e], linecolor=:black, label=nothing)
            end
            plot!([NaN], [NaN], linecolor=:black, label="plastic response active")

            p5 = plot()
            plot!(ts_output, X2_22s, label="\$(X_2)_{22}\$ [MPa]")
            plot!(ts_output, X2_11s, label="\$(X_2)_{11}\$ [MPa]")
            for (s, e) in runs
                plot!(ts_output[s:e], X2_22s[s:e], linecolor=:black, label=nothing)
            end
            plot!([NaN], [NaN], linecolor=:black, label="plastic response active")

            p6 = plot()
            plot!(ts_output, X3_22s, label="\$(X_3)_{22}\$ [MPa]")
            plot!(ts_output, X3_11s, label="\$(X_3)_{11}\$ [MPa]")
            for (s, e) in runs
                plot!(ts_output[s:e], X3_22s[s:e], linecolor=:black, label=nothing)
            end
            plot!([NaN], [NaN], linecolor=:black, label="plastic response active")

            plot(p1, p2, p3, p4, p5, p6, layout=(2, 3))
        end
    end
end
