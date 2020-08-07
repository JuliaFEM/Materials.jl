# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Materials, Test

@testset "Test Materials.jl" begin
    @testset "test utilities" begin
        include("test_utilities.jl")
    end
    @testset "test perfect plastic uniaxial stress" begin
        include("test_perfectplastic.jl")
    end
    @testset "test perfect plastic pure shear" begin
        include("test_perfectplastic_shear.jl")
    end
    @testset "test chaboche uniaxial stress" begin
        include("test_chaboche.jl")
    end
    @testset "test chaboche pure shear" begin
        include("test_chaboche_shear.jl")
    end
    @testset "test uniaxial increment" begin
        include("test_uniaxial_increment.jl")
    end
    @testset "test biaxial increment" begin
        include("test_biaxial_increment.jl")
    end
    @testset "test stress-driven uniaxial increment" begin
        include("test_stress_driven_uniaxial_increment.jl")
    end
end
