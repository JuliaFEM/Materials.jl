# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Tensors

@testset "calculate response in 2D, plane stress" begin
    E = 200e3
    nu = 0.3
    D = E/(1.0 - nu^2) .* [
                1.0  nu 0.0
                nu  1.0 0.0
                0.0 0.0 (1.0-nu)/2.0]
    gradu = [0.125 0.125;
             0.0   0.0]
    strain = 1/2*(gradu + gradu')
    strain_vec = [strain[1,1]; strain[2,2]; strain[1,2]]
    stress_vec = D * ([1.0, 1.0, 2.0] .* strain_vec)
    expected_stress_tensor = Tensor{1, 3}(stress_vec)
    mat = Material(2, formulation=:plane_stress)
    elastic = IsotropicHooke(200e3, 0.3)
    add_property!(mat, elastic, "elastic")
    t = 1.0
    stress_tensor = calc_response!(mat, gradu, t)
    @test expected_stress_tensor == stress_tensor
end  


@testset "calculate response in 2D, plane strain" begin
    E = 200e3
    nu = 0.3
    D = E/((1.0+nu)*(1.0-2.0*nu)) .* [
            1.0-nu      nu               0.0
                nu  1.0-nu               0.0
                0.0     0.0  (1.0-2.0*nu)/2.0]
    gradu = [0.125 0.125;
             0.0   0.0]
    strain = 1/2*(gradu + gradu')
    strain_vec = [strain[1,1]; strain[2,2]; strain[1,2]]
    stress_vec = D * ([1.0, 1.0, 2.0] .* strain_vec)
    expected_stress_tensor = Tensor{1, 3}(stress_vec)

    mat = Material(2, formulation=:plane_strain)
    elastic = IsotropicHooke(200e3, 0.3)
    add_property!(mat, elastic, "elastic")

    t = 1.0
    stress_tensor = calc_response!(mat, gradu, t)
    @test expected_stress_tensor == stress_tensor
end 

@testset "calculate response in 3D" begin
    # Expected value
    E = 200e3
    nu = 0.3
    gradu = [0.025 0.025 0.0;
             0.0   0.0   0.0;
             0.0   0.0   0.0]
    D = E/((1.0+nu)*(1.0-2.0*nu)) * [
            1.0-nu nu nu 0.0 0.0 0.0
            nu 1.0-nu nu 0.0 0.0 0.0
            nu nu 1.0-nu 0.0 0.0 0.0
            0.0 0.0 0.0 0.5-nu 0.0 0.0
            0.0 0.0 0.0 0.0 0.5-nu 0.0
            0.0 0.0 0.0 0.0 0.0 0.5-nu]

    strain = 1/2*(gradu' + gradu)
    strain_vec = [strain[1,1]; strain[2,2]; strain[3,3]; strain[1,2]; strain[2,3]; strain[1,3]]
    stress_vec = D * ([1.0, 1.0, 1.0, 2.0, 2.0, 2.0].*strain_vec)

    expected_stress_tensor = Tensor{2, 3}([ stress_vec[1] stress_vec[4] stress_vec[5];
                                            stress_vec[4] stress_vec[2] stress_vec[6];
                                            stress_vec[5] stress_vec[6] stress_vec[3]])

    mat = Material(3)
    elastic = IsotropicHooke(E, nu)
    add_property!(mat, elastic, "elastic")
    
    t = 1.0
    stress_tensor = calc_response!(mat, gradu, t)
    @test expected_stress_tensor == stress_tensor
end