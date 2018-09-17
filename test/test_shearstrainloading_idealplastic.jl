# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using JuliaFEM, FEMBase, Materials, Test
using Materials: OneElementSimulator, ShearStrainLoading

# Define load profile
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

loading = ShearStrainLoading(times, loads)
# Initialize material and simulator
mat = Material(IdealPlastic, ())
sim = OneElementSimulator(mat, loading)
update!(sim.elements, "youngs modulus", E)
update!(sim.elements, "poissons ratio", nu)
update!(sim.elements, "yield stress", syield)

# Initialize simulator and run
Materials.initialize!(sim)
Materials.run!(sim)
for i in 1:length(times)
    @info "time = $(sim.times[i]), strain = $(sim.strains[i]), stress = $(sim.stresses[i])"
end
s31s = [s[6] for s in sim.stresses]
@test isapprox(s31s, [0.0, syield/sqrt(3.0), syield/sqrt(3.0), 0.0, -syield/sqrt(3.0)])
