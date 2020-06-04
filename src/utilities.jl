# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

"""Alias for symmetric tensor type of rank 2, dimension 3."""
const Symm2{T} = SymmetricTensor{2,3,T}

"""Alias for symmetric tensor type of rank 4, dimension 3."""
const Symm4{T} = SymmetricTensor{4,3,T}

# Various rank-4 unit tensors, for documentation.
# Let A be a rank-2 tensor, and I the rank-2 unit tensor.
#
# typ = Float64
# II = Symm4{typ}((i,j,k,l) -> delta(i,k)*delta(j,l))  # II : A = A
# IT = Symm4{typ}((i,j,k,l) -> delta(i,l)*delta(j,k))  # IT : A = transpose(A)
# IS = Symm4{typ}((i,j,k,l) -> 0.5*(II + IT))  # symmetric
# IA = Symm4{typ}((i,j,k,l) -> 0.5*(II - IT))  # screw-symmetric
# IV = 1.0/3.0 * Symm4{typ}((i,j,k,l) -> delta(i,j)*delta(k,l))  # volumetric, IV = (1/3) I ⊗ I
# ID = IS - IV  # deviatoric

"""
    lame(E, nu)

Convert the elastic parameters (E, nu) to the Lamé parameters (lambda, mu). Isotropic material.
"""
@inline function lame(E::Real, nu::Real)
    lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lambda, mu
end

"""
    delame(lambda, mu)

Convert the Lamé parameters (lambda, mu) to the elastic parameters (E, nu). Isotropic material.
"""
@inline function delame(lambda::Real, mu::Real)
    E = mu * (3.0 * lambda + 2.0 * mu) / (lambda + mu)
    nu = lambda / (2.0 * (lambda + mu))
    return E, nu
end

"""
    isotropic_elasticity_tensor(lambda, mu)

Compute the elasticity tensor (rank 4, symmetric) for an isotropic material
having the Lamé parameters `lambda` and `mu`.

If you have (E, nu) instead, use `lame` to get (lambda, mu).
"""
function isotropic_elasticity_tensor(lambda::Real, mu::Real)
    delta(i,j) = i==j ? 1.0 : 0.0
    C(i,j,k,l) = lambda*delta(i,j)*delta(k,l) + mu*(delta(i,k)*delta(j,l) + delta(i,l)*delta(j,k))
    return Symm4{typeof(lambda)}(C)
end

"""
    debang(f!, ex=nothing)

Take away the bang; i.e. convert a mutating function into non-mutating form.

`f!` must be a two-argument mutating function, which writes the result into its
first argument. The result of `debang` is then `f`, a single-argument
non-mutating function that allocates and returns the result. Schematically,
`f!(out, x)` becomes `f(x)`.

When the type, size and shape of `out` is the same as those of `x`, it is enough
to supply just `f!`. When `f` is called, output will be allocated as `similar(x)`.

When the type, size and/or shape of `out` are different from those of `x`, then
an example instance of the correct type with the correct size and shape for the
output must be supplied, as debang's `ex` argument. When `f` is called, output
will be allocated as `similar(ex)`.

# Note

While the type of F is known at compile time, the size and shape are typically
runtime properties, not encoded into the type. For example, arrays have the
number of dimensions as part of the type, but the length of each dimension
is only defined at run time, when an instance is created. This is why the `ex`
argument is needed.

# Etymology

By convention, mutating functions are marked with an exclamation mark, a.k.a.
bang. Debanging takes away the bang.
"""
function debang(f!, ex=nothing)
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

# This comes from viscoplastic.jl, and is currently unused.
# The original error message suggested this was planned to be used for "radial return".
"""A simple Newton solver for x* such that f(x*) = 0."""
function find_root(f::Function, x::AbstractVector{<:Real},
            dfdx::Union{Function, Nothing}=nothing;
            max_iter::Integer=50, norm_acc::Real=1e-9)
    if dfdx === nothing
        dfdx = (x) -> ForwardDiff.jacobian(f, x)
    end
    converged = false
    for i=1:max_iter
        dx = -dfdx(x) \ f(x)
        x += dx
        if norm(dx) < norm_acc
            converged = true
            break
        end
    end
    converged || error("No convergence!")
    return x
end
