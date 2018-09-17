# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using JuliaFEM, FEMBase, Materials, Test
using Materials: OneElementSimulator, AxialStrainLoading

# Define load profile
times = [0.0]
loads = [0.0]
dt = 0.5
ea = 1.0e-3
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

loading = AxialStrainLoading(times, loads)
# Initialize material and simulator
mat = Material(IdealPlastic, ())
sim = OneElementSimulator(mat, loading)
update!(sim.elements, "youngs modulus", 200.0e3)
update!(sim.elements, "poissons ratio", 0.3)
update!(sim.elements, "yield stress", 100.0)

# Initialize simulator and run
Materials.initialize!(sim)
Materials.run!(sim)

s33s = [s[3] for s in sim.stresses]
@test isapprox(s33s, [0.0, 100.0, 100.0, 0.0, -100.0])
