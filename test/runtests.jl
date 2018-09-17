# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using FEMBase, Materials, Test

@testset "Test Materials.jl" begin
    @testset "test abstract material" begin
        include("test_abstract_material.jl")
    end
    @testset "test ideal plastic material model" begin
        include("test_idealplastic.jl")
    end
    @testset "test chaboche material model" begin
        include("test_chaboche.jl")
    end
    @testset "test simulator" begin
        include("test_simulator.jl")
    end
    @testset "test one element simulator" begin
        include("test_oneelementsimulator.jl")
    end
    @testset "test ideal plastic pure shear" begin
        include("test_shearstrainloading_idealplastic.jl")
    end
end
