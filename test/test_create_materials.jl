# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md
using Materials

@testset "Create material" begin
    mat = Material(1)
    @test mat.dimension == 1
    @test mat.finite_strain == false
end

@testset "Add property" begin
    mat = Material(3)
    elastic = IsotropicHooke(200e3, 0.3)
    add_property!(mat, elastic, "elastic")
    @test mat.dimension == 3
    @test mat.finite_strain == false
    @test mat.properties["elastic"] == elastic
end

@testset "Add property 2." begin
    mat = Material(3)
    add_property!(mat, IsotropicHooke, "elastic", 200e3, 0.3)
    @test mat.dimension == 3
    @test mat.finite_strain == false

    elastic = mat.properties["elastic"]
    @test isapprox(elastic.youngs_modulus, 200e3)
    @test isapprox(elastic.nu, 0.3)
end