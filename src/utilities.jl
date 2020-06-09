# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

"""Alias for symmetric tensor type of rank 2, dimension 3."""
const Symm2{T} = SymmetricTensor{2,3,T}

"""Alias for symmetric tensor type of rank 4, dimension 3."""
const Symm4{T} = SymmetricTensor{4,3,T}

"""Kronecker delta. delta(i, j) = (i == j) ? 1 : 0."""
delta(i::Integer, j::Integer) = (i == j) ? 1 : 0

"""Rank-4 unit tensor, defined by II : A = A for any rank-2 tensor A."""
II(T::Type=Float64) = Symm4{T}((i,j,k,l) -> delta(i,k)*delta(j,l))
"""Rank-4 unit tensor, defined by IT : A = transpose(A) for any rank-2 tensor A."""
IT(T::Type=Float64) = Symm4{T}((i,j,k,l) -> delta(i,l)*delta(j,k))

"""Rank-4 unit tensor, symmetric. IS ≡ (1/2) (II + IT)."""
IS(T::Type=Float64) = Symm4{T}((i,j,k,l) -> 1//2 * (II(T) + IT(T)))
"""Rank-4 unit tensor, screw-symmetric. IA ≡ (1/2) (II - IT)."""
IA(T::Type=Float64) = Symm4{T}((i,j,k,l) -> 1//2 * (II(T) - IT(T)))

"""Rank-4 unit tensor, volumetric. IS ≡ (1/3) I ⊗ I, where I is the rank-2 unit tensor."""
IV(T::Type=Float64) = Symm4{T}((i,j,k,l) -> 1//3 * delta(i,j)*delta(k,l))
"""Rank-4 unit tensor, deviatoric. ID ≡ IS - IV."""
ID(T::Type=Float64) = IS(T) - IV(T)

"""
    isotropic_elasticity_tensor(lambda, mu)

Compute the elasticity tensor C(i,j,k,l) (rank 4, symmetric) for an isotropic
material having the Lamé parameters `lambda` and `mu`.

If you have (E, nu) instead, use `lame` to get (lambda, mu).
"""
isotropic_elasticity_tensor(lambda::T, mu::T) where T <: Real = Symm4{T}((i,j,k,l) -> 3 * lambda * IV(T) + 2 * mu * IS(T))

"""
    lame(E, nu)

Convert elastic parameters (E, nu) of an isotropic material to Lamé parameters (lambda, mu).
"""
function lame(E::Real, nu::Real)
    lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lambda, mu
end

"""
    delame(lambda, mu)

Convert Lamé parameters (lambda, mu) of an isotropic material to elastic parameters (E, nu).
"""
function delame(lambda::Real, mu::Real)
    E = mu * (3 * lambda + 2 * mu) / (lambda + mu)
    nu = lambda / (2 * (lambda + mu))
    return E, nu
end

"""
    debang(f!, ex=nothing)

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
    else
        function f(x)
            out = similar(ex)
            f!(out, x)
            return out
        end
    end
    return f
end

# This comes from the old viscoplastic.jl, and is currently unused.
# The original wording of the error message suggested this was planned to be used for "radial return".
"""A simple Newton solver for x* such that f(x*) = 0.

The input `x` is the initial guess.

The default `dfdx=nothing` uses `ForwardDiff.jacobian` to compute the jacobian
automatically.

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
