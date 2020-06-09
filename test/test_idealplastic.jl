# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Test, Tensors

parameters = IdealPlasticParameterState(youngs_modulus=200.0e3,
                                        poissons_ratio=0.3,
                                        yield_stress=100.0)

dstrain_dtime = fromvoigt(Symm2{Float64}, 1e-3*[1.0, -0.3, -0.3, 0.0, 0.0, 0.0]; offdiagscale=2.0)
ddrivers = IdealPlasticDriverState(time=0.25, strain=0.25*dstrain_dtime)
mat = IdealPlastic(parameters=parameters, ddrivers=ddrivers)
integrate_material!(mat)
update_material!(mat)
@test isapprox(mat.variables.stress, fromvoigt(Symm2, [50.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

mat.ddrivers = ddrivers
integrate_material!(mat)
update_material!(mat)
@test isapprox(mat.variables.stress, fromvoigt(Symm2, [100.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
@test isapprox(mat.variables.cumeq, 0.0; atol=1.0e-12)

dstrain_dtime = fromvoigt(Symm2{Float64}, 1e-3*[1.0, -0.5, -0.5, 0.0, 0.0, 0.0]; offdiagscale=2.0)
ddrivers = IdealPlasticDriverState(time=0.25, strain=0.25*dstrain_dtime)
mat.ddrivers = ddrivers
integrate_material!(mat)
update_material!(mat)
@test isapprox(mat.variables.stress, fromvoigt(Symm2, [100.0, 0.0, 0.0, 0.0, 0.0, 0.0]); atol=1.0e-12)
@test isapprox(mat.variables.cumeq, 0.25*1.0e-3)

dstrain_dtime = fromvoigt(Symm2{Float64}, -1e-3*[1.0, -0.3, -0.3, 0.0, 0.0, 0.0]; offdiagscale=2.0)
ddrivers = IdealPlasticDriverState(time=0.25, strain=0.25*dstrain_dtime)
mat.ddrivers = ddrivers
integrate_material!(mat)
update_material!(mat)
@test isapprox(mat.variables.stress, fromvoigt(Symm2, [50.0, 0.0, 0.0, 0.0, 0.0, 0.0]); atol=1.0e-12)

dstrain_dtime = (-0.75*fromvoigt(Symm2{Float64}, 1e-3*[1.0, -0.3, -0.3, 0.0, 0.0, 0.0]; offdiagscale=2.0)
                 -0.25*fromvoigt(Symm2{Float64}, 1e-3*[1.0, -0.5, -0.5, 0.0, 0.0, 0.0]; offdiagscale=2.0))
ddrivers = IdealPlasticDriverState(time=1.0, strain=dstrain_dtime)
mat.ddrivers = ddrivers
integrate_material!(mat)
integrate_material!(mat)
update_material!(mat)
@test isapprox(mat.variables.stress, fromvoigt(Symm2, [-100.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
