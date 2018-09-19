# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using FEMMaterials, Test, ForwardDiff


@testset "Calculate plastic strain" begin
    dummyel = Element(Hex8, (1, 2, 3, 4, 5, 6, 7, 8))
    dummyip = first(get_integration_points(dummyel))
    # time = [...]
    # strain = [...]
    # stress = run_simulation(mat, time, strain)
    # @test isapprox(stress, stress_expected)

    n = 1.0
    K = 100.0

    mat = Material(ViscoPlastic, (:norton, [K, n]))
    mat.properties.youngs_modulus = 200.0e3
    mat.properties.poissons_ratio = 0.3
    mat.properties.yield_stress = 100.0

    mat.dtime = 0.25
    mat.dstrain .= 1.0e-3*[1.0, -0.3, -0.3, 0.0, 0.0, 0.0]*mat.dtime
    integrate_material!(mat)
    @test isapprox(mat.stress+mat.dstress, [50.0, 0.0, 00.0, 0.0, 0.0, 0.0])

    material_postprocess_increment!(mat, dummyel, dummyip, mat.time+mat.dtime)

    integrate_material!(mat)
    @test isapprox(mat.stress+mat.dstress, [100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    @test isapprox(mat.properties.dplastic_multiplier, 0.0; atol=1.0e-12)

    material_postprocess_increment!(mat, dummyel, dummyip, mat.time+mat.dtime)

    integrate_material!(mat)
    @test mat.properties.dplastic_strain[1] > 0
    @test mat.properties.dplastic_strain[2] < 0
    @test mat.properties.dplastic_strain[3] < 0

    stress = mat.stress
    dstress = mat.dstress

    @test Materials.von_mises_yield(stress + dstress, 200) <= 0
end

@testset "Invalid potential is not defined" begin
    @test_throws ErrorException Material(ViscoPlastic, (:notNorton, [0.0, 0.0]))
end
