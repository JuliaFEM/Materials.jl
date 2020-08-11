# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Test, Tensors, LinearAlgebra

# Kronecker delta
@test delta(1, 1) == 1
@test delta(1, 2) == 0
@test_throws MethodError delta(1.0, 2.0)  # indices must be integers
@test_throws MethodError delta(1, BigInt(2))  # both must have the same type
@test delta(1, 2) isa Int  # the output type matches the input
@test delta(BigInt(1), BigInt(2)) isa BigInt

# Various tensors
let Z3 = zeros(3, 3),
    O3 = ones(3, 3),
    eye(n) = Diagonal([1 for _ in range(1, length=n)]),  # TODO: in Julia 1.3 and later, we can just say I(n)
    I3 = eye(3)
    @test isapprox(tovoigt(II()), eye(6))
    @test isapprox(tovoigt(IT()), [I3  Z3;
                                   Z3  Z3])

    @test isapprox(tovoigt(IS()), [I3  Z3;
                                   Z3  1//2*I3])
    @test isapprox(tovoigt(IA()), [Z3  Z3;
                                   Z3  1//2*I3])

    @test isapprox(tovoigt(IV()), [1//3*O3  Z3;
                                   Z3       Z3])
    @test isapprox(tovoigt(ID()), [(I3 - 1//3*O3)  Z3;
                                   Z3              1//2*I3])

    @test let lambda = 10.0,
              mu = 1.0
        isapprox(tovoigt(isotropic_elasticity_tensor(lambda, mu)), [(lambda*O3 + 2*mu*I3)  Z3;
                                                              Z3               mu*I3])
    end
end

# Lamé parameters for isotropic solids
@test all(isapprox(result, expected)
        for (result, expected) in zip(lame(1e11, 0.3), (5.769230769230769e10, 3.846153846153846e10)))
@test all(isapprox(result, expected)
        for (result, expected) in zip(delame(lame(1e11, 0.3)...), (1e11, 0.3)))

# Mutating function to non-mutating function conversion
let  # introduce a local scope so the name `f!` is only defined locally for this test.
    function f!(out, x)
        out[:] = [sin(elt) for elt in x]
        return nothing
    end

    let
        out = [0.0]
        @test all([f!(out, [pi/4]) == nothing,
                 isapprox(out, [1/sqrt(2)])])
    end

    let
        out = [0.0]
        f = debang(f!)
        @test f isa Function
        @test all([isapprox(f([pi/4]), [1/sqrt(2)]),
                 out == [0.0]])
    end
end

# Newton root finder
let g(x) = [(1 - x[1]^2) + x[2]],
    x0 = [0.8, 0.2]
    @test !isapprox(g(x0), [0.0], atol=1e-15)  # find_root should have to actually do something
    @test isapprox(g(find_root(g, x0)), [0.0], atol=1e-15)
end
