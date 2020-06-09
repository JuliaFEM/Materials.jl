# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Test, Tensors, LinearAlgebra

@test delta(1, 2) isa Int64
@test delta(BigInt(1), BigInt(2)) isa BigInt
@test_throws MethodError delta(1.0, 2.0)

@test isapprox(tovoigt(II()), I(6))
@test let Z = zeros(3, 3)
        isapprox(tovoigt(IT()), Array([I(3) Z;
                                       Z    Z]))
      end

@test let Z = zeros(3, 3)
    isapprox(tovoigt(IS()), Array([I(3) Z;
                                   Z    1//2*I(3)]))
end
@test let Z = zeros(3, 3)
    isapprox(tovoigt(IA()), Array([Z  Z;
                                   Z  1//2*I(3)]))
end

@test let Z = zeros(3, 3),
          O = ones(3, 3)
    isapprox(tovoigt(IV()), Array([1//3*O Z;
                                   Z      Z]))
end
@test let Z = zeros(3, 3)
    isapprox(tovoigt(ID()), Array([  2//3  -1//3  -1//3  0     0     0;
                                    -1//3   2//3  -1//3  0     0     0;
                                    -1//3  -1//3   2//3  0     0     0;
                                     0      0      0     1//2  0     0;
                                     0      0      0     0     1//2  0;
                                     0      0      0     0     0     1//2]))
end

@test isapprox(tovoigt(isotropic_elasticity_tensor(10.0, 0.3)), [10.6  10.0  10.0  0.0  0.0  0.0;
                                                                 10.0  10.6  10.0  0.0  0.0  0.0;
                                                                 10.0  10.0  10.6  0.0  0.0  0.0;
                                                                  0.0   0.0   0.0  0.3  0.0  0.0;
                                                                  0.0   0.0   0.0  0.0  0.3  0.0;
                                                                  0.0   0.0   0.0  0.0  0.0  0.3])

@test all(isapprox(a, b) for (a, b) in zip(lame(1e11, 0.3), (5.769230769230769e10, 3.846153846153846e10)))
@test all(isapprox(a, b) for (a, b) in zip(delame(lame(1e11, 0.3)...), (1e11, 0.3)))

function test_debang()  # just to introduce a local scope so the names `f!`, `f` and `out` are local to this test
    function f!(out, x)
        out[:] = [sin(elt) for elt in x]
        return nothing
    end
    out = [0.0]
    @test all([f!(out, [pi/4]) == nothing,
             isapprox(out[1], 1/sqrt(2))])

    f = debang(f!)
    @test f isa Function
    out = [0.0]
    @test all([isapprox(f([pi/4])[1], 1/sqrt(2)),
             out[1] == 0.0])
end
test_debang()

# The output of g must be an AbstractArray to use ForwardDiff.jacobian (the default `dfdx`) in find_root.
let g(x) = [(1 - x[1]^2) + x[2]],
    x0 = [0.8, 0.2]
    @test !isapprox(g(x0), [0.0], atol=1e-15)  # find_root should have to actually do something
    @test isapprox(g(find_root(g, x0)), [0.0], atol=1e-15)
end
