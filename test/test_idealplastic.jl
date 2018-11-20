# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Test, Tensors
# time = [...]
# strain = [...]
# stress = run_simulation(mat, time, strain)
# @test isapprox(stress, stress_expected)

#
# mat = Material(IdealPlastic, tuple())
# mat.properties.youngs_modulus = 200.0e3
# mat.properties.poissons_ratio = 0.3
# mat.properties.yield_stress = 100.0

parameters = IdealPlasticParameterState(youngs_modulus = 200.0e3,
                                        poissons_ratio = 0.3,
                                        yield_stress = 100.0)

dstrain_dtime = fromvoigt(SymmetricTensor{2,3,Float64},1e-3*[1.0, -0.3, -0.3, 0.0, 0.0, 0.0]; offdiagscale=2.0)
ddrivers = IdealPlasticDriverState(time = 0.25, strain = 0.25*dstrain_dtime)
mat = IdealPlastic(parameters=parameters, ddrivers=ddrivers)
# mat.dtime = 0.25
# mat.dstrain .= 1.0e-3*[-0.3, -0.3, 1.0, 0.0, 0.0, 0.0]*mat.dtime
integrate_material!(mat)
update_material!(mat)
@test isapprox(mat.variables.stress, fromvoigt(SymmetricTensor{2,3}, [50.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

# mat.time += mat.dtime
# mat.strain .+= mat.dstrain
# mat.stress .+= mat.dstress
mat.ddrivers = ddrivers
integrate_material!(mat)
update_material!(mat)
@test isapprox(mat.variables.stress, fromvoigt(SymmetricTensor{2,3}, [100.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
@test isapprox(mat.variables.cumeq, 0.0; atol=1.0e-12)

# mat.time += mat.dtime
# mat.strain .+= mat.dstrain
# mat.stress .+= mat.dstress
# mat.dstrain[:] .= 1.0e-3*[-0.5, -0.5, 1.0, 0.0, 0.0, 0.0]*mat.dtime
dstrain_dtime = fromvoigt(SymmetricTensor{2,3,Float64}, 1e-3*[1.0, -0.5, -0.5, 0.0, 0.0, 0.0]; offdiagscale=2.0)
ddrivers = IdealPlasticDriverState(time = 0.25, strain = 0.25*dstrain_dtime)
mat.ddrivers = ddrivers
integrate_material!(mat)
update_material!(mat)
@test isapprox(mat.variables.stress, fromvoigt(SymmetricTensor{2,3}, [100.0, 0.0, 0.0, 0.0, 0.0, 0.0]); atol=1.0e-12)
@test isapprox(mat.variables.cumeq, 0.25*1.0e-3)

# mat.time += mat.dtime
# mat.strain .+= mat.dstrain
# mat.stress .+= mat.dstress
# mat.dstrain[:] .= -1.0e-3*[-0.3, -0.3, 1.0, 0.0, 0.0, 0.0]*mat.dtime
dstrain_dtime = fromvoigt(SymmetricTensor{2,3,Float64}, -1e-3*[1.0, -0.3, -0.3, 0.0, 0.0, 0.0]; offdiagscale=2.0)
ddrivers = IdealPlasticDriverState(time = 0.25, strain = 0.25*dstrain_dtime)
mat.ddrivers = ddrivers
integrate_material!(mat)
update_material!(mat)
@test isapprox(mat.variables.stress, fromvoigt(SymmetricTensor{2,3}, [50.0, 0.0, 0.0, 0.0, 0.0, 0.0]); atol=1.0e-12)

# mat.time += mat.dtime
# mat.strain .+= mat.dstrain
# mat.stress .+= mat.dstress
# mat.dtime = 1.0
# m1 = 1.0e-3*[-0.3, -0.3, 1.0, 0.0, 0.0, 0.0]
# m2 = 1.0e-3*[-0.5, -0.5, 1.0, 0.0, 0.0, 0.0]
# mat.dstrain[:] .= -m1*0.75 - m2*0.25
dstrain_dtime = (-0.75*fromvoigt(SymmetricTensor{2,3,Float64}, 1e-3*[1.0, -0.3, -0.3, 0.0, 0.0, 0.0]; offdiagscale=2.0)
                -0.25*fromvoigt(SymmetricTensor{2,3,Float64}, 1e-3*[1.0, -0.5, -0.5, 0.0, 0.0, 0.0]; offdiagscale=2.0))
ddrivers = IdealPlasticDriverState(time = 1.0, strain = dstrain_dtime)
mat.ddrivers = ddrivers
integrate_material!(mat)
integrate_material!(mat)
update_material!(mat)
@test isapprox(mat.variables.stress, fromvoigt(SymmetricTensor{2,3}, [-100.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
# @test isapprox(mat.properties.dplastic_multiplier, 0.25*1.0e-3)
