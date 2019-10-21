# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Test, Tensors

# mat = Material(Chaboche, tuple())
# props = mat.properties
# props.youngs_modulus = 200.0e3
# props.poissons_ratio = 0.3
# props.yield_stress = 100.0
# props.K_n = 100.0
# props.n_n = 3.0
# props.C_1 = 0.0
# props.D_1 = 100.0
# props.C_2 = 0.0
# props.D_2 = 1000.0
# props.Q = 0.0
# props.b = 0.1
E = 200.0e3
nu = 0.3
parameters = ChabocheParameterState(E = E,
                                    nu = nu,
                                    R0 = 100.0,
                                    Kn = 100.0,
                                    nn = 3.0,
                                    C1 = 0.0,
                                    D1 = 100.0,
                                    C2 = 0.0,
                                    D2 = 1000.0,
                                    Q = 0.0,
                                    b = 0.1)
mat = Chaboche(parameters = parameters)
times = [0.0]
loads = [0.0]
dt = 0.5
G = 0.5*E/(1+nu)
syield = 100.0
# vm = sqrt(3)*G*ga | ea = ga
ea = 2*syield/(sqrt(3)*G)
# Go to elastic border
push!(times, times[end]+dt)
push!(loads, loads[end] + ea*dt)
 # Proceed to plastic flow
push!(times, times[end]+dt)
push!(loads, loads[end] + ea*dt)
 # Reverse direction
push!(times, times[end]+dt)
push!(loads, loads[end] - ea*dt)
 # Continue and pass yield criterion
push!(times, times[end]+dt)
push!(loads, loads[end] - ea*dt)
push!(times, times[end]+dt)
push!(loads, loads[end] - ea*dt)

eeqs = [mat.variables.cumeq]
stresses = [copy(tovoigt(mat.variables.stress))]
for i=2:length(times)
    dtime = times[i]-times[i-1]
    dstrain31 = loads[i]-loads[i-1]
    dstrain = [0.0, 0.0, 0.0, 0.0, 0.0, dstrain31]
    # mat.dtime = dtime
    # mat.dstrain = dstrain
    dstrain_ = fromvoigt(SymmetricTensor{2,3,Float64}, dstrain; offdiagscale=2.0)
    mat.ddrivers = ChabocheDriverState(time = dtime, strain = dstrain_)
    integrate_material!(mat)
    # mat.time += mat.dtime
    # mat.strain .+= mat.dstrain
    # mat.stress .+= mat.dstress
    # mat.properties.plastic_strain .+= mat.properties.dplastic_strain
    # mat.properties.backstress1 .+= mat.properties.dbackstress1
    # mat.properties.backstress2 .+= mat.properties.dbackstress2
    # mat.properties.R += mat.properties.dR
    update_material!(mat)
    push!(stresses, copy(tovoigt(mat.variables.stress)))
    push!(eeqs, mat.variables.cumeq)
    # @info "time = $(mat.time), stress = $(mat.stress), cumeq = $(mat.properties.cumulative_equivalent_plastic_strain))"
end

for i in 1:length(times)
    @test isapprox(stresses[i][1:5], zeros(5); atol=1e-6)
end
s31 = [s[6] for s in stresses]

@test isapprox(s31[2], syield/sqrt(3.0))
@test isapprox(s31[3]*sqrt(3.0), syield + 100.0*((eeqs[3]-eeqs[2])/dt)^(1.0/3.0); rtol=1e-2)
@test isapprox(s31[4], s31[3]-G*ea*dt)
@test isapprox(s31[6]*sqrt(3.0), -(syield + 100.0*((eeqs[6]-eeqs[5])/dt)^(1.0/3.0)); rtol=1e-2)
