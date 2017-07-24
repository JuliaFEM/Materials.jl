# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md
using Base.Test
using Materials
using Tensors

@testset "Plasticity: Verify Von Mises yield function, positive output" begin
    stress_tensor = Tensor{2, 3}([200.   0. 0.;
                                    0. 200. 0.;
                                    0.   0. 0.])
    stress_tensor = yield_function(stress_tensor, 150., Val{:VonMises})
    @test isapprox(stress_tensor, 50.)
end

@testset "Plasticity: Verify Von Mises yield function, negative output" begin
    stress_tensor = Tensor{2, 3}([100.   0. 0.;
                                    0. 100. 0.;
                                    0.   0. 0.])
    stress_tensor = yield_function(stress_tensor, 150., Val{:VonMises})
    @test isapprox(stress_tensor, -50.)
end

@testset "Plasticity: Calculate radial return with Von Mises" begin
    E = 200.0
    nu = 0.3
    elastic = IsotropicHooke(E, nu)
    vonmises = VonMises(200.0)
    gradu = [1.25 0.025 0.0;
            0.0   0.0   0.24;
            0.0   0.24   0.0]
    # Calculation
    model = Model(elastic, vonmises)
    mat = Materials.create_material(3, model)

    calc_response!(mat, gradu)
    stress = mat.trial_values["stress"]
    yields = Materials.yield_function(stress, vonmises)
    @test abs(yields - 200.0) < 1e5
end


@testset "Plasticity: Elastic response #TODO" begin
    E = 200.0
    nu = 0.3
    elastic = IsotropicHooke(E, nu)
    vonmises = VonMises(200.0)
    gradu = [1.25 0.025 0.0;
            0.0   0.0   0.24;
            0.0   0.24   0.0]
    # Calculation
    model = Model(elastic)
    mat = Materials.create_material(3, model)

    calc_response!(mat, gradu)
    @test 1==1
end

@testset "Plasticity: Elastic response finite strain #TODO" begin
    E = 200.0
    nu = 0.3
    elastic = IsotropicHooke(E, nu)
    vonmises = VonMises(200.0)
    gradu = [1.25 0.025 0.0;
            0.0   0.0   0.24;
            0.0   0.24   0.0]
    # Calculation
    model = Model(elastic)
    mat = Materials.create_material(3, model, finite_strain=true)

    calc_response!(mat, gradu)
    @test 1==1
end