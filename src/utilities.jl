# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

module Utilities

using Tensors, ForwardDiff

export Symm2, Symm4
export delta, II, IT, IS, IA, IV, ID, isotropic_elasticity_tensor, isotropic_compliance_tensor
export lame, delame, debang, find_root

"""Symm2{T} is an alias for SymmetricTensor{2,3,T}."""
const Symm2{T} = SymmetricTensor{2,3,T}

"""Symm4{T} is an alias for SymmetricTensor{4,3,T}."""
const Symm4{T} = SymmetricTensor{4,3,T}

"""
    delta(i::Integer, j::Integer)

Kronecker delta, defined by delta(i, j) = (i == j) ? 1 : 0.
"""
delta(i::T, j::T) where T <: Integer = (i == j) ? one(T) : zero(T)

# TODO: We could probably remove the type argument, and just let the results be
# inferred as Symm4{Int64}, Symm4{Rational{Int64}} and similar. Simpler to use,
# and those behave correctly in calculations with types involving other reals
# such as Float64.
# Performance implications? Is the Julia compiler smart enough to optimize?
"""
    II(T::Type=Float64)

Rank-4 unit tensor, defined by II : A = A for any rank-2 tensor A.
"""
II(T::Type=Float64) = Symm4{T}((i,j,k,l) -> delta(i,k)*delta(j,l))

"""
    IT(T::Type=Float64)

Rank-4 unit tensor, defined by IT : A = transpose(A) for any rank-2 tensor A.
"""
IT(T::Type=Float64) = Symm4{T}((i,j,k,l) -> delta(i,l)*delta(j,k))

"""
    IS(T::Type=Float64)

Rank-4 unit tensor, symmetric. IS ≡ (1/2) (II + IT).
"""
IS(T::Type=Float64) = 1//2 * (II(T) + IT(T))

"""
    IA(T::Type=Float64)

Rank-4 unit tensor, screw-symmetric. IA ≡ (1/2) (II - IT).
"""
IA(T::Type=Float64) = 1//2 * (II(T) - IT(T))

"""
    IV(T::Type=Float64)

Rank-4 unit tensor, volumetric. IS ≡ (1/3) I ⊗ I, where I is the rank-2 unit tensor.
"""
IV(T::Type=Float64) = Symm4{T}((i,j,k,l) -> 1//3 * delta(i,j)*delta(k,l))

"""
    ID(T::Type=Float64)

Rank-4 unit tensor, deviatoric. ID ≡ IS - IV.
"""
ID(T::Type=Float64) = IS(T) - IV(T)

"""
    isotropic_elasticity_tensor(lambda::T, mu::T) where T <: Real

Compute the elasticity tensor C(i,j,k,l) (rank 4, symmetric) for an isotropic
material having the Lamé parameters `lambda` and `mu`.

If you have (E, nu) instead, use `lame` to get (lambda, mu).
"""
isotropic_elasticity_tensor(lambda::T, mu::T) where T <: Real = 3 * lambda * IV(T) + 2 * mu * IS(T)

# TODO: check: original expr from upstream/master:
#     g(i,j,k,l) = -lambda/(2.0*mu*(3.0*lambda + 2.0*mu))*delta(i,j)*delta(k,l) + 1.0/(4.0*mu)*(delta(i,k)*delta(j,l)+delta(i,l)*delta(j,k))
"""
    isotropic_compliance_tensor(lambda::T, mu::T) where T <: Real

Compute the compliance tensor S(i,j,k,l) (rank 4, symmetric) for an isotropic
material having the Lamé parameters `lambda` and `mu`.

If you have (E, nu) instead, use `lame` to get (lambda, mu).
"""
isotropic_compliance_tensor(lambda::T, mu::T) where T <: Real = -3 * lambda / (2*mu * (3*lambda + 2*mu)) * IV(T) + 1 / (2*mu) * IS(T)

"""
    lame(E::Real, nu::Real)

Convert elastic parameters (E, nu) of an isotropic material to Lamé parameters (lambda, mu).

See:
    https://en.wikipedia.org/wiki/Template:Elastic_moduli
"""
function lame(E::Real, nu::Real)
    lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lambda, mu
end

"""
    delame(lambda::Real, mu::Real)

Convert Lamé parameters (lambda, mu) of an isotropic material to elastic parameters (E, nu).

See:
    https://en.wikipedia.org/wiki/Template:Elastic_moduli
"""
function delame(lambda::Real, mu::Real)
    E = mu * (3 * lambda + 2 * mu) / (lambda + mu)
    nu = lambda / (2 * (lambda + mu))
    return E, nu
end

"""
    debang(f!::Function, ex=nothing)

Convert a mutating function into non-mutating form.

`f!` must be a two-argument mutating function, which writes the result into its
first argument. The result of `debang` is then `f`, a single-argument
non-mutating function that allocates and returns the result. Schematically,
`f!(out, x)` becomes `f(x)`.

When the type, size and shape of `out` is the same as those of `x`, it is enough
to supply just `f!`. When `f` is called, output will be allocated as `similar(x)`.

When the type, size and/or shape of `out` are different from those of `x`, then
an example instance of the correct type with the correct size and shape for the
output must be supplied, as debang's `ex` argument. When `f` is called, output
will be allocated as `similar(ex)`. The `ex` instance will be automatically kept
alive by the lexical closure of `f`.

# Note

While the type of `out` is known at compile time, the size and shape are
typically runtime properties, not encoded into the type. For example, arrays
have the number of dimensions as part of the type, but the length of each
dimension is only defined at run time, when an instance is created. This is why
the `ex` argument is needed.

# Etymology

By convention, mutating functions are marked with an exclamation mark, a.k.a.
bang. This function takes away the bang.
"""
function debang(f!::Function, ex=nothing)
    if ex === nothing
        function f(x)
            out = similar(x)
            f!(out, x)
            return out
        end
        return f
    else
        # We use a different name to make incremental compilation happy.
        function f_with_ex(x)
            out = similar(ex)
            f!(out, x)
            return out
        end
        return f_with_ex
    end
end

# This comes from the old viscoplastic.jl, and is currently unused.
# The original wording of the error message suggested this was planned to be used for "radial return".
"""A simple Newton solver for the vector x* such that f(x*) = 0.

The input `x` is the initial guess.

The default `dfdx=nothing` uses `ForwardDiff.jacobian` to compute the jacobian
automatically. In this case the output of `f` must be an `AbstractArray`.

`tol` is measured in the vector norm of the change in `x` between successive
iterations.
"""
function find_root(f::Function, x::AbstractVector{<:Real},
                   dfdx::Union{Function, Nothing}=nothing;
                   max_iter::Integer=50, tol::Real=1e-9)
    if dfdx === nothing
        dfdx = (x) -> ForwardDiff.jacobian(f, x)
    end
    for i=1:max_iter
        dx = -dfdx(x) \ f(x)
        x += dx
        if norm(dx) < tol
            return x
        end
    end
    error("No convergence!")
end

end
