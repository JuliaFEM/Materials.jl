# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Tensors

# See Hooke's law definitions from: https://en.wikipedia.org/wiki/Hooke's_law

@testset "calculate response in 2D, plane stress" begin
    # Material definitions
    E = 200e3
    nu = 0.3
    G = E / (2*(1 + nu))

    # nabla u = dx/dX
    gradu = [0.125 0.125;
             0.45   0.01]
    
    # Small strains
    strain = 1/2*(gradu + gradu')
    strain_vec = [strain[1,1]; strain[2,2]; strain[1,2]]

    # Analytical stresses
    sigma_x = E / (1-nu^2) * (strain_vec[1] + nu * strain_vec[2])
    sigma_y = E / (1 - nu^2) * (strain_vec[2] + nu * strain_vec[1])
    sigma_xy = G * 2 * strain_vec[3]
    expected = [ sigma_x sigma_xy 0;
                sigma_xy  sigma_y 0;
                       0        0 0]

    expected_stress_tensor = Tensor{2, 3}(expected)

    # Calculation
    mat = Material(2, formulation=:plane_stress)
    elastic = IsotropicHooke(200e3, 0.3)
    add_property!(mat, elastic, "elastic")

    stress_tensor = calc_response(mat, gradu)
    @test isapprox(expected_stress_tensor, stress_tensor)
end

@testset "calculate response in 2D, plane strain" begin
    # Material definitions
    E = 200e3
    nu = 0.3

    # nabla u = dx/dX
    gradu = [0.125 0.125;
             0.0   0.0]

    # Small strains
    strain = 1/2*(gradu + gradu')
    strain_vec = [strain[1,1]; strain[2,2]; strain[1,2]]
    G = E / (2*(1 + nu))

    # Analytical stresses
    sigma_x = 2*G / (1 - 2*nu) * ((1-nu) * strain_vec[1] + nu*strain_vec[2])
    sigma_y = 2*G / (1 - 2*nu) * ((1-nu) * strain_vec[2] + nu*strain_vec[1])
    sigma_z = 2*nu*G / (1 - 2*nu) * (strain_vec[1] + nu*strain_vec[2])
    sigma_xy = G *(strain_vec[1] + strain_vec[2])
    expected = [ sigma_x sigma_xy       0;
                sigma_xy  sigma_y       0;
                       0        0 sigma_z]
    expected_stress_tensor = Tensor{2, 3}(expected)

    # Calculation
    mat = Material(2, formulation=:plane_strain)
    elastic = IsotropicHooke(200e3, 0.3)
    add_property!(mat, elastic, "elastic")

    stress_tensor = calc_response(mat, gradu)
    @test isapprox(expected_stress_tensor, stress_tensor)
end

@testset "calculate response in 3D" begin
    # Material definitions
    E = 200e3
    nu = 0.3

    # nabla u = dx/dX
    gradu = [0.025 0.025 0.0;
             0.0   0.0   0.24;
             0.0   0.24   0.0]

    # Elastic moduli
    D = E/((1.0+nu)*(1.0-2.0*nu)) * [
            1.0-nu nu nu 0.0 0.0 0.0
            nu 1.0-nu nu 0.0 0.0 0.0
            nu nu 1.0-nu 0.0 0.0 0.0
            0.0 0.0 0.0 0.5-nu 0.0 0.0
            0.0 0.0 0.0 0.0 0.5-nu 0.0
            0.0 0.0 0.0 0.0 0.0 0.5-nu]

    # Small strains
    strain = 1/2*(gradu' + gradu)
    strain_vec = [strain[1,1]; strain[2,2]; strain[3,3]; strain[1,2]; strain[2,3]; strain[1,3]]
    stress_vec = D * ([1.0, 1.0, 1.0, 2.0, 2.0, 2.0].*strain_vec)

    expected_stress_tensor = Tensor{2, 3}([ stress_vec[1] stress_vec[4] stress_vec[6];
                                            stress_vec[4] stress_vec[2] stress_vec[5];
                                            stress_vec[6] stress_vec[5] stress_vec[3]])

    # Calculation
    mat = Material(3)
    elastic = IsotropicHooke(E, nu)
    add_property!(mat, elastic, "elastic")

    stress_tensor = calc_response(mat, gradu)
    @test expected_stress_tensor == stress_tensor
end
