# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md
using Materials
using Base.Test

@testset "Create material: Elastic material" begin
    elastic = IsotropicHooke(200e3, 0.3)
    model = Model(elastic)
    mat = create_material(3, model)
    @test mat.dimension == 3
    @test mat.finite_strain == false
    model_ = mat.model
    elastic_ = model_.elastic
    @test isapprox(elastic_.youngs_modulus, 200e3)
    @test isapprox(elastic_.nu, 0.3)
end