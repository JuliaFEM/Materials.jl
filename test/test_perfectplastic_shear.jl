# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Test, Tensors

let E = 200.0e3,
    nu = 0.3,
    yield_strength = 100.0,
    parameters = PerfectPlasticParameterState(youngs_modulus=E,
                                              poissons_ratio=nu,
                                              yield_stress=yield_strength),  # yield in shear = R0 / sqrt(3)
    mat = PerfectPlastic(parameters=parameters),
    times = [0.0],
    loads = [0.0],
    dt = 1.0,
    G = 0.5*E/(1+nu),
    # vonMises = sqrt(3 J_2) = sqrt(3/2 tr(s^2)) = sqrt(3) |tau| = sqrt(3)*G*|gamma|
    # gamma = 2 e12
    # set  vonMises = Y
    gamma_yield = yield_strength/(sqrt(3)*G)

    # Go to elastic border
    push!(times, times[end] + dt)
    push!(loads, loads[end] + gamma_yield*dt)
    # Proceed to plastic flow
    push!(times, times[end] + dt)
    push!(loads, loads[end] + gamma_yield*dt)
    # Reverse direction
    push!(times, times[end] + dt)
    push!(loads, loads[end] - gamma_yield*dt)
    # Continue and pass yield criterion
    push!(times, times[end] + dt)
    push!(loads, loads[end] - 2*gamma_yield*dt)

    stresses = [copy(tovoigt(mat.variables.stress))]
    for i=2:length(times)
        dtime = times[i] - times[i-1]
        dstrain31 = loads[i] - loads[i-1]
        dstrain_voigt = [0.0, 0.0, 0.0, 0.0, 0.0, dstrain31]
        dstrain_tensor = fromvoigt(Symm2{Float64}, dstrain_voigt; offdiagscale=2.0)
        mat.ddrivers = PerfectPlasticDriverState(time=dtime, strain=dstrain_tensor)
        integrate_material!(mat)
        update_material!(mat)
        push!(stresses, copy(tovoigt(mat.variables.stress)))
    end

    for i in 1:length(times)
        @test isapprox(stresses[i][1:5], zeros(5); atol=1e-6)
    end

    let y = yield_strength/sqrt(3.0)
        s31 = [s[6] for s in stresses]
        s31_expected = [0.0, y, y, 0.0, -y]
        @test isapprox(s31, s31_expected; rtol=1.0e-2)
    end
end
