# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Test, Tensors

let dtime = 0.25,
    R0 = 100.0,
    parameters = ChabocheParameterState(E=200.0e3,
                                        nu=0.3,
                                        R0=R0,
                                        Kn=100.0,
                                        nn=10.0,
                                        C1=10000.0,
                                        D1=100.0,
                                        C2=50000.0,
                                        D2=1000.0,
                                        Q=0.0,
                                        b=0.1),
    material = Chaboche(parameters=parameters),
    times = [material.drivers.time],
    stresses = [copy(tovoigt(material.variables.stress))],
    strains = [copy(tovoigt(material.drivers.strain; offdiagscale=2.0))],
    cumeqs = [copy(material.variables.cumeq)],
    tostrain(vec) = fromvoigt(Symm2, vec; offdiagscale=2.0),
    tostress(vec) = fromvoigt(Symm2, vec),
    uniaxial_stress(sigma) = tostress([sigma, 0, 0, 0, 0, 0]),
    stresses_expected = [uniaxial_stress(R0 / 2),
                         uniaxial_stress(R0),
                         uniaxial_stress(1.5 * R0),
                         uniaxial_stress(1.5 * R0),
                         uniaxial_stress(R0),
                         uniaxial_stress(-R0)],
    dstress = R0 / 2,
    dstresses11 = dstress*[1.0, 1.0, 1.0, 0.0, -1.0, -4.0]
    dtimes = [dtime, dtime, dtime, 1e3, dtime, 1e3]

    for i in 1:length(dtimes)
        dstress11 = dstresses11[i]
        dtime = dtimes[i]
        stress_driven_uniaxial_increment!(material, dstress11, dtime)
        update_material!(material)
        push!(times, material.drivers.time)
        push!(stresses, copy(tovoigt(material.variables.stress)))
        push!(strains, copy(tovoigt(material.drivers.strain; offdiagscale=2.0)))
        push!(cumeqs, copy(material.variables.cumeq))
        @test isapprox(material.variables.stress, stresses_expected[i]; atol=1e-4)
    end

    # Plastic creep should have occurred at the portion of the test
    # where the stress was held at 1.5*R0.
    dstrain_creep = strains[5] - strains[4]
    # von Mises material, so the plastic nu = 0.5.
    @test isapprox(dstrain_creep[2], -dstrain_creep[1]*0.5; atol=1e-4)  # ε22 = -0.5 ε11
    @test isapprox(dstrain_creep[3], -dstrain_creep[1]*0.5; atol=1e-4)  # ε33 = -0.5 ε11

    dcumeq = cumeqs[end] - cumeqs[end-1]
    @test dcumeq > 0
end
