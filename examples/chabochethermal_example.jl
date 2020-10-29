# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Tensors
using Plots
using Materials
pyplot()

let
    function constant(value::Real)
        function interpolate(x::Real)
            return value
        end
        return interpolate
    end

    function capped_linear(x1::Real, y1::Real, x2::Real, y2::Real)
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
        T1 = K(600.0),
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
                                                   nu=capped_linear(T0, 0.3, T1, 0.35),
                                                   alpha=capped_linear(T0, 1.216e-5, T1, 1.680e-5),
                                                   R0=capped_linear(T0, 100.0, T1, 50.0),
                                                   # viscous hardening in constant strain rate test: (tvp * ε')^(1/nn) * Kn
                                                   tvp=1000.0,
                                                   Kn=capped_linear(T0, 100.0, T1, 50.0),
                                                   nn=capped_linear(T0, 1.0, T1, 4.0),
                                                   # C1=constant(10000.0),
                                                   # D1=constant(100.0),
                                                   # C2=constant(50000.0),
                                                   # D2=constant(1000.0),
                                                   C1=constant(0.0),
                                                   D1=constant(0.0),
                                                   C2=constant(0.0),
                                                   D2=constant(0.0),
                                                   C3=constant(0.0),
                                                   D3=constant(0.0),
                                                   Q=capped_linear(T0, 50.0, T1, 10.0),
                                                   b=capped_linear(T0, 100.0, T1, 0.01)),
                                                   # Q=constant(0.0),
                                                   # b=constant(0.0)),
        # uniaxial pull test, so we set only dε11.
        strain_rate=1e-4,  # dε/dt [1/s]
        strain_final=0.005,  # when to stop the pull test
        dt=0.25,  # simulation timestep, [s]
        dstrain11 = strain_rate * dt,  # dε11 during one timestep
        n_timesteps = Integer(round(strain_final / dstrain11)),
        constant_temperatures = range(T0, T1, length=3),
        timevar_temperature = range(T0, T0 + 130, length=n_timesteps + 1)

        plot()  # make empty figure

        println("Constant temperature tests")
        for T in constant_temperatures
            println("T = $(degreesC(T))°C")
            mat = ChabocheThermal(parameters=parameters)
            mat.drivers.temperature = T
            mat.ddrivers.temperature = 0
            stresses = [mat.variables.stress[1,1]]
            strains = [mat.drivers.strain[1,1]]
            for i in 1:n_timesteps
                uniaxial_increment!(mat, dstrain11, dt)
                update_material!(mat)
                push!(strains, mat.drivers.strain[1,1])
                push!(stresses, mat.variables.stress[1,1])
            end
            println("    ε11, σ11, at end of simulation")
            println("    $(strains[end]), $(stresses[end])")
            plot!(strains, stresses, label="\$\\sigma(\\varepsilon)\$ @ \$$(degreesC(T))°C\$")
        end

        println("Time-varying temperature tests (activates ΔT terms)")
        println("T = $(degreesC(timevar_temperature[1]))°C ... $(degreesC(timevar_temperature[end]))°C, linear profile.")
        mat = ChabocheThermal(parameters=parameters)
        stresses = [mat.variables.stress[1,1]]
        strains = [mat.drivers.strain[1,1]]
        for (Ta, Tb) in zip(timevar_temperature, timevar_temperature[2:end])
            # println("        Ta = $(degreesC(Ta))°C, Tb = $(degreesC(Tb))°C, ΔT = $(Tb - Ta)°C")
            mat.drivers.temperature = Tb
            mat.ddrivers.temperature = Tb - Ta
            uniaxial_increment!(mat, dstrain11, dt)
            update_material!(mat)
            push!(strains, mat.drivers.strain[1,1])
            push!(stresses, mat.variables.stress[1,1])
        end
        println("    ε11, σ11, at end of simulation")
        println("    $(strains[end]), $(stresses[end])")
        plot!(strains, stresses, label="\$\\sigma(\\varepsilon)\$ @ $(degreesC(timevar_temperature[1]))°C ... $(degreesC(timevar_temperature[end]))°C")

        # plot!(xx2, yy2, label="...")  # to add new curves into the same figure
        xlabel!("\$\\varepsilon\$")
        ylabel!("\$\\sigma\$")
        title!("Stress-strain response")
    end
end
