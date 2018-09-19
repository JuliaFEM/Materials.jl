# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Materials, Test

mat = Material(Chaboche, tuple())
props = mat.properties
props.youngs_modulus = 200.0e3
props.poissons_ratio = 0.3
props.yield_stress = 100.0
props.K_n = 10.0
props.n_n = 20.0
props.C_1 = 0.0
props.D_1 = 100.0
props.C_2 = 0.0
props.D_2 = 1000.0
props.Q = 0.0
props.b = 0.1

times = [0.0]
loads = [0.0]
dt = 0.5
E = 200.0e3
nu = 0.3
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
push!(loads, loads[end] - 2*ea*dt)

eeqs = [mat.properties.cumulative_equivalent_plastic_strain]
stresses = [copy(mat.stress)]
for i=2:length(times)
    dtime = times[i]-times[i-1]
    dstrain31 = loads[i]-loads[i-1]
    dstrain = [0.0, 0.0, 0.0, 0.0, 0.0, dstrain31]
    mat.dtime = dtime
    mat.dstrain = dstrain
    integrate_material!(mat)
    mat.time += mat.dtime
    mat.strain .+= mat.dstrain
    mat.stress .+= mat.dstress
    mat.properties.plastic_strain .+= mat.properties.dplastic_strain
    mat.properties.backstress1 .+= mat.properties.dbackstress1
    mat.properties.backstress2 .+= mat.properties.dbackstress2
    mat.properties.R += mat.properties.dR
    mat.properties.cumulative_equivalent_plastic_strain += mat.properties.dcumulative_equivalent_plastic_strain
    push!(stresses, copy(mat.stress))
    push!(eeqs, mat.properties.cumulative_equivalent_plastic_strain)
end

for (i) in 1:length(times)
    @test isapprox(stresses[i][1:5], zeros(5); atol=1e-6)
end
s31 = [s[6] for s in stresses]

@test isapprox(s31[2], syield/sqrt(3.0))
@test isapprox(s31[3]*sqrt(3.0), syield + 10.0*((eeqs[3]-eeqs[2])/dt)^(1.0/20.0); rtol=1e-2)
@test isapprox(s31[4], s31[3]-G*ea*dt)
@test isapprox(s31[5]*sqrt(3.0), -(syield + 10.0*((eeqs[5]-eeqs[4])/dt)^(1.0/20.0)); rtol=1e-2)
