# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using FEMBase, Materials, Test

@testset "Test Materials.jl" begin
    @testset "test abstract material" begin
        include("test_abstract_material.jl")
    end
end
