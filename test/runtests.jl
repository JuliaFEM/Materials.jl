# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Materials, Test

@testset "Test Materials.jl" begin
    # @testset "test abstract material" begin
    #     include("test_abstract_material.jl")
    # end
    @testset "test ideal plastic material model" begin
        include("test_idealplastic.jl")
    end
    @testset "test ideal plastic pure shear" begin
        include("test_idealplastic_shear.jl")
    end
    @testset "test uniaxial increment" begin
        include("test_uniaxial_increment.jl")
    end
    @testset "test chaboche pure shear" begin
        include("test_chaboche_shear.jl")
    end
    @testset "test chaboche uniaxial stress" begin
        include("test_chaboche.jl")
    end
    @testset "test biaxial increment" begin
        include("test_biaxial_increment.jl")
    end
    @testset "test memory" begin
        include("test_memory.jl")
    end
    @testset "test stress-driven uniaxial increment" begin
        include("test_stress_driven_uniaxial_increment.jl")
    end
    @testset "test DSA material model" begin
        include("test_DSA.jl")
    end
    # @testset "test viscoplastic material model" begin
    #     include("test_viscoplastic.jl")
    # end
end
