# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Test, Tensors

let dtime = 0.25,
    nu = 0.3,
    R = 100.0,
    parameters = PerfectPlasticParameterState(youngs_modulus=200.0e3,
                                              poissons_ratio=nu,
                                              yield_stress=R),
    mat = PerfectPlastic(parameters=parameters),
    tostrain(vec) = fromvoigt(Symm2, vec; offdiagscale=2.0),
    tostress(vec) = fromvoigt(Symm2, vec),
    uniaxial_stress(sigma) = tostress([sigma, 0, 0, 0, 0, 0]),
    stresses_expected = [uniaxial_stress(R / 2),
                         uniaxial_stress(R),
                         uniaxial_stress(R),
                         uniaxial_stress(R / 2),
                         uniaxial_stress(-R)],
    dstrain11 = 1e-3*dtime,
    strains_expected = [tostrain(dstrain11*[1.0, -nu, -nu, 0.0, 0.0, 0.0]),
                        tostrain(dstrain11*[2, -2*nu, -2*nu, 0.0, 0.0, 0.0]),
                        tostrain(dstrain11*[3, -2*nu - 0.5, -2*nu - 0.5, 0.0, 0.0, 0.0]),
                        tostrain(dstrain11*[2, -nu - 0.5, -nu - 0.5, 0.0, 0.0, 0.0]),
                        tostrain(dstrain11*[-2, 2*nu, 2*nu, 0.0, 0.0, 0.0])],
    dtimes = [dtime, dtime, dtime, dtime, 1.0],
    dstrains11 = dstrain11*[1.0, 1.0, 1.0, -1.0, -4.0]

    for i in 1:length(dtimes)
        dstrain11 = dstrains11[i]
        dtime = dtimes[i]
        uniaxial_increment!(mat, dstrain11, dtime)
        update_material!(mat)
        @test isapprox(mat.variables.stress, stresses_expected[i])
        @test isapprox(mat.drivers.strain, strains_expected[i])
    end
end
