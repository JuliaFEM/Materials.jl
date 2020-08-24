# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Test, Tensors

let E = 200.0e3,
    nu = 0.3,
    yield_strength = 100.0,
    parameters = ChabocheParameterState(E=E,
                                        nu=nu,
                                        R0=yield_strength,  # yield in shear = R0 / sqrt(3)
                                        Kn=100.0,
                                        nn=3.0,
                                        C1=0.0,
                                        D1=100.0,
                                        C2=0.0,
                                        D2=1000.0,
                                        Q=0.0,
                                        b=0.1),
    mat = Chaboche(parameters=parameters),
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
    push!(loads, loads[end] - gamma_yield*dt)
    push!(times, times[end] + dt)
    push!(loads, loads[end] - gamma_yield*dt)

    eeqs = [mat.variables.cumeq]
    stresses = [copy(tovoigt(mat.variables.stress))]
    for i=2:length(times)
        dtime = times[i] - times[i-1]
        dstrain12 = loads[i] - loads[i-1]
        dstrain_voigt = [0.0, 0.0, 0.0, 0.0, 0.0, dstrain12]
        dstrain_tensor = fromvoigt(Symm2{Float64}, dstrain_voigt; offdiagscale=2.0)
        mat.ddrivers = ChabocheDriverState(time=dtime, strain=dstrain_tensor)
        integrate_material!(mat)
        # @info "$i, $gamma_yield, $(mat.variables_new.stress[1,2]), $(2.0*mat.variables_new.plastic_strain[1,2])\n"
        update_material!(mat)
        push!(stresses, copy(tovoigt(mat.variables.stress)))
        push!(eeqs, mat.variables.cumeq)
        # @info "time = $(mat.time), stress = $(mat.stress), cumeq = $(mat.properties.cumulative_equivalent_plastic_strain))"
    end

    for i in 1:length(times)
        @test isapprox(stresses[i][1:5], zeros(5); atol=1e-6)
    end

    s31 = [s[6] for s in stresses]
    @test isapprox(s31[2], yield_strength/sqrt(3.0))
    @test isapprox(s31[3]*sqrt(3.0), yield_strength + 100.0*((eeqs[3] - eeqs[2])/dt)^(1.0/3.0); rtol=1e-2)
    @test isapprox(s31[4], s31[3] - G*gamma_yield*dt)
    @test isapprox(s31[6]*sqrt(3.0), -(yield_strength + 100.0*((eeqs[6] - eeqs[5])/dt)^(1.0/3.0)); rtol=1e-2)
end
