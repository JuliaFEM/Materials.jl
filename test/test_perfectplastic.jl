# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Test, Tensors

let nu = 0.3,
    yield_strength=100.0,
    parameters = PerfectPlasticParameterState(youngs_modulus=200.0e3,
                                              poissons_ratio=nu,
                                              yield_stress=yield_strength),
    epsilon=1e-3,
    mat,  # scope the name to this level; actual definition follows later
    tostrain(vec) = fromvoigt(Symm2, vec; offdiagscale=2.0),
    tostress(vec) = fromvoigt(Symm2, vec),
    uniaxial_stress(sigma) = tostress([sigma, 0, 0, 0, 0, 0])
    let dtime=0.25
        # elastic straining
        dstrain_dtime = tostrain(epsilon*[1.0, -nu, -nu, 0.0, 0.0, 0.0])
        ddrivers = PerfectPlasticDriverState(time=dtime, strain=dstrain_dtime*dtime)
        mat = PerfectPlastic(parameters=parameters, ddrivers=ddrivers)
        integrate_material!(mat)
        update_material!(mat)
        @test isapprox(mat.variables.stress, uniaxial_stress(yield_strength / 2))

        mat.ddrivers = ddrivers
        integrate_material!(mat)
        update_material!(mat)
        # We should now be at the yield surface.
        @test isapprox(mat.variables.stress, uniaxial_stress(yield_strength))
        @test isapprox(mat.variables.cumeq, 0.0; atol=1.0e-12)

        # plastic straining
        # von Mises material, plastically incompressible, so plastic nu=0.5.
        dstrain_dtime = tostrain(epsilon*[1.0, -0.5, -0.5, 0.0, 0.0, 0.0])
        ddrivers = PerfectPlasticDriverState(time=dtime, strain=dstrain_dtime*dtime)
        mat.ddrivers = ddrivers
        integrate_material!(mat)
        update_material!(mat)
        @test isapprox(mat.variables.stress, uniaxial_stress(yield_strength); atol=1.0e-12)
        @test isapprox(mat.variables.cumeq, dtime*epsilon)

        # return to elastic state
        dstrain_dtime = tostrain(-epsilon*[1.0, -nu, -nu, 0.0, 0.0, 0.0])
        ddrivers = PerfectPlasticDriverState(time=dtime, strain=dstrain_dtime*dtime)
        mat.ddrivers = ddrivers
        integrate_material!(mat)
        update_material!(mat)
        @test isapprox(mat.variables.stress, uniaxial_stress(yield_strength / 2); atol=1.0e-12)
    end

    let dtime=1.0
        # loading in reverse direction to plastic state
        # The 0.75 term: one 0.25 cancels the current elastic stress state,
        # and the remaining 0.5 reaches the yield surface.
        # The 0.25 term: plastic strain.
        dstrain_dtime = (-0.75*tostrain(epsilon*[1.0, -nu, -nu, 0.0, 0.0, 0.0])
                         -0.25*tostrain(epsilon*[1.0, -0.5, -0.5, 0.0, 0.0, 0.0]))
        ddrivers = PerfectPlasticDriverState(time=1.0, strain=dstrain_dtime*dtime)
        mat.ddrivers = ddrivers
        integrate_material!(mat)
        integrate_material!(mat)
        update_material!(mat)
        @test isapprox(mat.variables.stress, uniaxial_stress(-yield_strength))
    end
end
