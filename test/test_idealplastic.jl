# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Test
# time = [...]
# strain = [...]
# stress = run_simulation(mat, time, strain)
# @test isapprox(stress, stress_expected)

mat = Material(IdealPlastic, tuple())
mat.properties.youngs_modulus = 200.0e3
mat.properties.poissons_ratio = 0.3
mat.properties.yield_stress = 100.0

mat.dtime = 0.25
mat.dstrain .= 1.0e-3*[-0.3, -0.3, 1.0, 0.0, 0.0, 0.0]*mat.dtime
integrate_material!(mat)
@test isapprox(mat.stress+mat.dstress, [0.0, 0.0, 50.0, 0.0, 0.0, 0.0])

mat.time += mat.dtime
mat.strain .+= mat.dstrain
mat.stress .+= mat.dstress

integrate_material!(mat)
@test isapprox(mat.stress+mat.dstress, [0.0, 0.0, 100.0, 0.0, 0.0, 0.0])
@test isapprox(mat.properties.dplastic_multiplier, 0.0; atol=1.0e-12)

mat.time += mat.dtime
mat.strain .+= mat.dstrain
mat.stress .+= mat.dstress
mat.dstrain[:] .= 1.0e-3*[-0.5, -0.5, 1.0, 0.0, 0.0, 0.0]*mat.dtime
integrate_material!(mat)
@test isapprox(mat.dstress, zeros(6); atol=1.0e-12)
@test isapprox(mat.properties.dplastic_multiplier, mat.dtime*1.0e-3)

mat.time += mat.dtime
mat.strain .+= mat.dstrain
mat.stress .+= mat.dstress
mat.dstrain[:] .= -1.0e-3*[-0.3, -0.3, 1.0, 0.0, 0.0, 0.0]*mat.dtime
integrate_material!(mat)
@test isapprox(mat.dstress, [0.0, 0.0, -50.0, 0.0, 0.0, 0.0]; atol=1.0e-12)

mat.time += mat.dtime
mat.strain .+= mat.dstrain
mat.stress .+= mat.dstress
mat.dtime = 1.0
m1 = 1.0e-3*[-0.3, -0.3, 1.0, 0.0, 0.0, 0.0]
m2 = 1.0e-3*[-0.5, -0.5, 1.0, 0.0, 0.0, 0.0]
mat.dstrain[:] .= -m1*0.75 - m2*0.25
integrate_material!(mat)
@test isapprox(mat.stress+mat.dstress, [0.0, 0.0, -100.0, 0.0, 0.0, 0.0])
@test isapprox(mat.properties.dplastic_multiplier, 0.25*1.0e-3)
