# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using Materials
using ForwardDiff
using Tensors
using Base.Test

@testset "Solvers: find root, approach positive side" begin
    f(x) = x.^2 - 5
    df(x) = ForwardDiff.jacobian(f, x)
    x = vec([3])
    root = find_root(f, df, x; max_iter=50, norm_acc=1e-9)
    @test isapprox(root, [sqrt(5)])
end

@testset "Solvers: find root, approach positive side" begin
    f(x) = x.^2 - 5
    df(x) = ForwardDiff.jacobian(f, x)
    x = vec([1])
    root = find_root(f, df, x; max_iter=50, norm_acc=1e-9)
    @test isapprox(root, [sqrt(5)])
end

@testset "Solvers: find root, multiple inputs" begin
    f(x) = [x[1]^2 + x[2]^2 - 25.]
    df(x) = ForwardDiff.jacobian(f, x)
    x = vec([1, 3])
    root = find_root(f, df, x; max_iter=50, norm_acc=1e-9)
    value = abs(f(root)[1]) < 1e8
    @test value
end

@testset "Solvers: Radial return with Newton" begin
    vonmises = VonMises(200.0)
    yield_f(stress) = Materials.yield_function(stress, vonmises)
    d_yield_f(stress) = Materials.d_yield_function(stress, 1, Val{:VonMises})
    E = 200.0
    nu = 0.3
    D = Materials.calc_elastic_tensor(E, nu)

    stress = Tensor{2,3}([  205.0   0.0  0.0;
                              0.0 205.0  0.0;
                              0.0   0.0  0.0])


    dstrain = Tensor{2,3}([ 0.001 0.060 0.010;
                            0.060 0.000 0.020;
                            0.010 0.020 0.010])
    params = Dict{AbstractString, Any}()
    params["yield_function"] = yield_f
    params["init_stress"] = stress
    params["dstrain"] = dstrain
    params["d_yield_function"] = d_yield_f
    params["D"] = D
    x = zeros(7)
    x[1:6] += 0.1

    f(x) = radial_return(x, params)
    df(x) = ForwardDiff.jacobian(f, x)
    root = find_root(f, df, x; max_iter=50, norm_acc=1e-9)
    dstress = Materials.array_to_tensor(root)
    stress_new = stress + dstress
    @test abs(yield_f(stress_new) - 200.0) < 1e5
end
