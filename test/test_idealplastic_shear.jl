# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Materials, FEMBase, Test
include("FEMMaterials.jl")
using .FEMMaterials
include("MaterialSimulators.jl")
using .MaterialSimulators


analysis, problem, element, bc_elements, ip = get_one_element_material_analysis(:IdealPlastic)
update!(element, "youngs modulus", 200.0e3)
update!(element, "poissons ratio", 0.3)
update!(element, "yield stress", 100.0)

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
update_bc_elements!(bc_elements, loading)

analysis.properties.t1 = maximum(times)
run!(analysis)
s31 = [ip("stress", t)[6] for t in times]
s31_expected = [0.0, syield/sqrt(3.0), syield/sqrt(3.0), 0.0, -syield/sqrt(3.0)]
@test isapprox(s31, s31_expected; rtol=1.0e-2)
