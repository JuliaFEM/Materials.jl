# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Materials, Test

mat = Material(IdealPlastic, tuple())
mat.properties.youngs_modulus = 200.0e3
mat.properties.poissons_ratio = 0.3
mat.properties.yield_stress = 100.0

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
    push!(stresses, copy(mat.stress))
end

for i in 1:length(times)
    @test isapprox(stresses[i][1:5], zeros(5); atol=1e-6)
end
s31 = [s[6] for s in stresses]

s31_expected = [0.0, syield/sqrt(3.0), syield/sqrt(3.0), 0.0, -syield/sqrt(3.0)]
@test isapprox(s31, s31_expected; rtol=1.0e-2)
