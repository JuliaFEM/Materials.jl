# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Materials, Test
include("FEMMaterials.jl")
using .FEMMaterials
include("MaterialSimulators.jl")
using .MaterialSimulators


m1 = 1.0e-3*[-0.3, -0.3, 1.0, 0.0, 0.0, 0.0]
m2 = 1.0e-3*[-0.5, -0.5, 1.0, 0.0, 0.0, 0.0]
mat = Material(IdealPlastic, ())
mat.properties.youngs_modulus = 200.0e3
mat.properties.poissons_ratio = 0.3
mat.properties.yield_stress = 100.0
mat.stress = zeros(6)
times = [0.0]
strains = [zeros(6)]
dt = 0.5
# Go to elastic border
push!(times, times[end]+dt)
push!(strains, strains[end] + m1*dt)

# Proceed to plastic flow
push!(times, times[end]+dt)
push!(strains, strains[end] + m2*dt)

# Reverse direction
push!(times, times[end]+dt)
push!(strains, strains[end] - m1*dt)

# Continue and pass yield criterion
push!(times, times[end]+dt)
push!(strains, strains[end] - (m1 + m2)*dt)

sim = Simulator(mat)
MaterialSimulators.initialize!(sim, strains, times)
MaterialSimulators.run!(sim)

s33s = [s[3] for s in sim.stresses]
@test isapprox(s33s, [0.0, 100.0, 100.0, 0.0, -100.0])
