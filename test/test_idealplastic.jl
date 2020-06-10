# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Test, Tensors

let parameters = IdealPlasticParameterState(youngs_modulus=200.0e3,
                                            poissons_ratio=0.3,
                                            yield_stress=100.0),
    epsilon=1e-3,
    mat,  # scope the name to this level; actual definition follows later
    tostrain(vec) = fromvoigt(Symm2, vec; offdiagscale=2.0),
    tostress(vec) = fromvoigt(Symm2, vec),
    uniaxial_stress(sigma) = tostress([sigma, 0, 0, 0, 0, 0])
    let dtime=0.25
        dstrain_dtime = tostrain(epsilon*[1.0, -0.3, -0.3, 0.0, 0.0, 0.0])
        ddrivers = IdealPlasticDriverState(time=dtime, strain=dstrain_dtime*dtime)
        mat = IdealPlastic{Float64}(parameters=parameters, ddrivers=ddrivers)
        integrate_material!(mat)
        update_material!(mat)
        @test isapprox(mat.variables.stress, uniaxial_stress(50.0))

        mat.ddrivers = ddrivers
        integrate_material!(mat)
        update_material!(mat)
        @test isapprox(mat.variables.stress, uniaxial_stress(100.0))
        @test isapprox(mat.variables.cumeq, 0.0; atol=1.0e-12)

        dstrain_dtime = tostrain(epsilon*[1.0, -0.5, -0.5, 0.0, 0.0, 0.0])
        ddrivers = IdealPlasticDriverState(time=dtime, strain=dstrain_dtime*dtime)
        mat.ddrivers = ddrivers
        integrate_material!(mat)
        update_material!(mat)
        @test isapprox(mat.variables.stress, uniaxial_stress(100.0); atol=1.0e-12)
        @test isapprox(mat.variables.cumeq, dtime*epsilon)

        dstrain_dtime = tostrain(-epsilon*[1.0, -0.3, -0.3, 0.0, 0.0, 0.0])
        ddrivers = IdealPlasticDriverState(time=dtime, strain=dstrain_dtime*dtime)
        mat.ddrivers = ddrivers
        integrate_material!(mat)
        update_material!(mat)
        @test isapprox(mat.variables.stress, uniaxial_stress(50.0); atol=1.0e-12)
    end

    dstrain_dtime = (-0.75*tostrain(epsilon*[1.0, -0.3, -0.3, 0.0, 0.0, 0.0])
                     -0.25*tostrain(epsilon*[1.0, -0.5, -0.5, 0.0, 0.0, 0.0]))
    ddrivers = IdealPlasticDriverState(time=1.0, strain=dstrain_dtime*1.0)
    mat.ddrivers = ddrivers
    integrate_material!(mat)
    integrate_material!(mat)
    update_material!(mat)
    @test isapprox(mat.variables.stress, uniaxial_stress(-100.0))
end
