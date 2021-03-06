# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

module ChabocheModule

using LinearAlgebra, ForwardDiff, Tensors, NLsolve, Parameters

import ..AbstractMaterial, ..AbstractMaterialState
import ..Utilities: Symm2, Symm4, isotropic_elasticity_tensor, lame, debang
import ..integrate_material!  # for method extension

# parametrically polymorphic for any type representing ℝ
export GenericChaboche, GenericChabocheDriverState, GenericChabocheParameterState, GenericChabocheVariableState

# specialization for Float64
export Chaboche, ChabocheDriverState, ChabocheParameterState, ChabocheVariableState

@with_kw mutable struct GenericChabocheDriverState{T <: Real} <: AbstractMaterialState
    time::T = zero(T)
    strain::Symm2{T} = zero(Symm2{T})
end

"""Parameter state for Chaboche material.

The classical viscoplastic material is a special case of this model with `C1 = C2 = 0`.

- `E`: Young's modulus
- `nu`: Poisson's ratio
- `R0`: initial yield strength
- `Kn`: plasticity multiplier divisor (drag stress)
- `nn`: plasticity multiplier exponent
- `C1`, `D1`: parameters governing behavior of backstress X1
- `C2`, `D2`: parameters governing behavior of backstress X2
- `Q`: hardening saturation state
- `b`: rate of convergence to hardening saturation
"""
@with_kw struct GenericChabocheParameterState{T <: Real} <: AbstractMaterialState
    E::T = 0
    nu::T = 0
    R0::T = 0
    Kn::T = 0
    nn::T = 0
    C1::T = 0
    D1::T = 0
    C2::T = 0
    D2::T = 0
    Q::T = 0
    b::T = 0
end

"""Problem state for Chaboche material.

- `stress`: stress tensor
- `X1`: backstress 1
- `X2`: backstress 2
- `plastic_strain`: plastic part of strain tensor
- `cumeq`: cumulative equivalent plastic strain (scalar, ≥ 0)
- `R`: yield strength
- `jacobian`: ∂σij/∂εkl
"""
@with_kw struct GenericChabocheVariableState{T <: Real} <: AbstractMaterialState
    stress::Symm2{T} = zero(Symm2{T})
    X1::Symm2{T} = zero(Symm2{T})
    X2::Symm2{T} = zero(Symm2{T})
    plastic_strain::Symm2{T} = zero(Symm2{T})
    cumeq::T = zero(T)
    R::T = zero(T)
    jacobian::Symm4{T} = zero(Symm4{T})
end

# TODO: Does this eventually need a {T}?
@with_kw struct ChabocheOptions <: AbstractMaterialState
    nlsolve_method::Symbol = :trust_region
end

@with_kw mutable struct GenericChaboche{T <: Real} <: AbstractMaterial
    drivers::GenericChabocheDriverState{T} = GenericChabocheDriverState{T}()
    ddrivers::GenericChabocheDriverState{T} = GenericChabocheDriverState{T}()
    variables::GenericChabocheVariableState{T} = GenericChabocheVariableState{T}()
    variables_new::GenericChabocheVariableState{T} = GenericChabocheVariableState{T}()
    parameters::GenericChabocheParameterState{T} = GenericChabocheParameterState{T}()
    dparameters::GenericChabocheParameterState{T} = GenericChabocheParameterState{T}()
    options::ChabocheOptions = ChabocheOptions()
end

ChabocheDriverState = GenericChabocheDriverState{Float64}
ChabocheParameterState = GenericChabocheParameterState{Float64}
ChabocheVariableState = GenericChabocheVariableState{Float64}
Chaboche = GenericChaboche{Float64}

"""
    state_to_vector(sigma::U, R::T, X1::U, X2::U) where U <: Symm2{T} where T <: Real

Adaptor for `nlsolve`. Marshal the problem state into a `Vector`.
"""
function state_to_vector(sigma::U, R::T, X1::U, X2::U) where U <: Symm2{T} where T <: Real
    return [tovoigt(sigma); R; tovoigt(X1); tovoigt(X2)]::Vector{T}
end

"""
    state_from_vector(x::AbstractVector{<:Real})

Adaptor for `nlsolve`. Unmarshal the problem state from a `Vector`.
"""
function state_from_vector(x::AbstractVector{T}) where T <: Real
    sigma::Symm2{T} = fromvoigt(Symm2{T}, @view x[1:6])
    R::T = x[7]
    X1::Symm2{T} = fromvoigt(Symm2{T}, @view x[8:13])
    X2::Symm2{T} = fromvoigt(Symm2{T}, @view x[14:19])
    return sigma, R, X1, X2
end

"""
    integrate_material!(material::GenericChaboche{T}) where T <: Real

Chaboche material with two backstresses. Both kinematic and isotropic hardening.

See:

    J.-L. Chaboche. Constitutive equations for cyclic plasticity and cyclic
    viscoplasticity. International Journal of Plasticity 5(3) (1989), 247--302.
    https://doi.org/10.1016/0749-6419(89)90015-6

Further reading:

    J.-L. Chaboche. A review of some plasticity and viscoplasticity constitutive
    theories. International Journal of Plasticity 24 (2008), 1642--1693.
    https://dx.doi.org/10.1016/j.ijplas.2008.03.009

    J.-L. Chaboche, A. Gaubert, P. Kanouté, A. Longuet, F. Azzouz, M. Mazière.
    Viscoplastic constitutive equations of combustion chamber materials including
    cyclic hardening and dynamic strain aging. International Journal of Plasticity
    46 (2013), 1--22. https://dx.doi.org/10.1016/j.ijplas.2012.09.011
"""
function integrate_material!(material::GenericChaboche{T}) where T <: Real
    p = material.parameters
    v = material.variables
    dd = material.ddrivers
    d = material.drivers
    @unpack E, nu, R0, Kn, nn, C1, D1, C2, D2, Q, b = p
    lambda, mu = lame(E, nu)

    @unpack strain, time = d
    dstrain = dd.strain
    dtime = dd.time
    @unpack stress, X1, X2, plastic_strain, cumeq, R = v

    # elastic part
    jacobian = isotropic_elasticity_tensor(lambda, mu)  # dσ/dε, i.e. ∂σij/∂εkl
    stress += dcontract(jacobian, dstrain)  # add the elastic stress increment, get the elastic trial stress

    # resulting deviatoric plastic stress (accounting for backstresses Xm)
    seff_dev = dev(stress - X1 - X2)
    # von Mises yield function
    f = sqrt(1.5)*norm(seff_dev) - (R0 + R)  # using elastic trial problem state
    if f > 0.0
        g! = create_nonlinear_system_of_equations(material)
        x0 = state_to_vector(stress, R, X1, X2)
        res = nlsolve(g!, x0; method=material.options.nlsolve_method, autodiff=:forward)  # user manual: https://github.com/JuliaNLSolvers/NLsolve.jl
        converged(res) || error("Nonlinear system of equations did not converge!")
        x = res.zero
        stress, R, X1, X2 = state_from_vector(x)

        # using the new problem state
        seff_dev = dev(stress - X1 - X2)
        f = sqrt(1.5)*norm(seff_dev) - (R0 + R)

        dotp = ((f >= 0.0 ? f : 0.0)/Kn)^nn  # power law viscoplasticity (Norton-Bailey type)
        dp = dotp*dtime  # |dε_p|, using backward Euler (dotp is ∂ε_p/∂t at the end of the timestep)
        n = sqrt(1.5)*seff_dev/norm(seff_dev)  # Chaboche: a (tensorial) unit direction, s.t. 2/3 * (n : n) = 1; also n = ∂f/∂σ.

        plastic_strain += dp*n
        cumeq += dp   # cumulative equivalent plastic strain (note dp ≥ 0)

        # Compute the new Jacobian, accounting for the plastic contribution. Because
        #   x ≡ [σ R X1 X2]   (vector of length 19, with tensors encoded in Voigt format)
        # we have
        #   dσ/dε = (dx/dε)[1:6,1:6]
        # for which we can compute the LHS as follows:
        #   dx/dε = dx/dr dr/dε = inv(dr/dx) dr/dε ≡ (dr/dx) \ (dr/dε)
        # where r = r(x) is the residual, given by the function g!. AD can get us dr/dx automatically,
        # the other factor we will have to supply manually.
        drdx = ForwardDiff.jacobian(debang(g!), x)  # Array{19, 19}
        drde = zeros((length(x),6))                 # Array{19, 6}
        drde[1:6, 1:6] = tovoigt(jacobian)  # elastic Jacobian. Follows from the defn. of g!.
        jacobian = fromvoigt(Symm4, (drdx\drde)[1:6, 1:6])
    end
    variables_new = GenericChabocheVariableState{T}(stress = stress,
                                                    X1 = X1,
                                                    X2 = X2,
                                                    R = R,
                                                    plastic_strain = plastic_strain,
                                                    cumeq = cumeq,
                                                    jacobian = jacobian)
    material.variables_new = variables_new
    return nothing
end

"""
    create_nonlinear_system_of_equations(material::GenericChaboche{T}) where T <: Real

Create and return an instance of the equation system for the incremental form of
the evolution equations of the Chaboche material.

Used internally for computing the plastic contribution in `integrate_material!`.

The input `material` represents the problem state at the end of the previous
timestep. The created equation system will hold its own copy of that state.

The equation system is represented as a mutating function `g!` that computes the
residual:

```julia
    g!(F::V, x::V) where V <: AbstractVector{<:Real}
```

Both `F` (output) and `x` (input) are length-19 vectors containing
[sigma, R, X1, X2], in that order. The tensor quantities sigma, X1,
X2 are encoded in Voigt format.

The function `g!` is intended to be handed over to `nlsolve`.
"""
function create_nonlinear_system_of_equations(material::GenericChaboche{T}) where T <: Real
    p = material.parameters
    v = material.variables
    dd = material.ddrivers
    d = material.drivers
    @unpack E, nu, R0, Kn, nn, C1, D1, C2, D2, Q, b = p
    lambda, mu = lame(E, nu)

    # Old problem state (i.e. the problem state at the time when this equation
    # system instance was created).
    #
    # Note this does not include the elastic trial; this is the state at the
    # end of the previous timestep.
    @unpack strain, time = d
    dstrain = dd.strain
    dtime = dd.time
    @unpack stress, X1, X2, plastic_strain, cumeq, R = v

    jacobian = isotropic_elasticity_tensor(lambda, mu)

    # Compute the residual. F is output, x is filled by NLsolve.
    # The solution is x = x* such that g(x*) = 0.
    function g!(F::V, x::V) where V <: AbstractVector{<:Real}
        stress_new, R_new, X1_new, X2_new = state_from_vector(x)  # tentative new values from nlsolve

        seff_dev = dev(stress_new - X1_new - X2_new)
        f = sqrt(1.5)*norm(seff_dev) - (R0 + R_new)

        dotp = ((f >= 0.0 ? f : 0.0)/Kn)^nn
        dp = dotp*dtime
        n = sqrt(1.5)*seff_dev/norm(seff_dev)

        # The equations are written in an incremental form:
        #
        # Δσ = (∂σ/∂ε)_e : dε_e = (∂σ/∂ε)_e : (dε - dε_p)   (components 1:6)
        # ΔR = b (Q - R_new) |dε_p|                         (component 7)
        # ΔX1 = (2/3) C1 |dε_p| (n - (3/2) (D1/C1) X1_new)  (components 8:13)
        # ΔX2 = (2/3) C2 |dε_p| (n - (3/2) (D2/C2) X2_new)  (components 14:19)
        #
        # where
        #
        # Δ(...) = (...)_new - (...)_old
        #
        # Then move the terms on the RHS to the LHS to get the standard form, (stuff) = 0.
        # Also, below we avoid the multiplication and division that cancel each other
        # in the last terms of the equations for ΔX1 and ΔX2.
        #
        dstrain_plastic = dp*n
        dstrain_elastic = dstrain - dstrain_plastic
        tovoigt!(view(F, 1:6), stress_new - stress - dcontract(jacobian, dstrain_elastic))

        F[7] = R_new - R - b*(Q - R_new)*dp

        tovoigt!(view(F,  8:13), X1_new - X1 - dp*(2.0/3.0*C1*n - D1*X1_new))
        tovoigt!(view(F, 14:19), X2_new - X2 - dp*(2.0/3.0*C2*n - D2*X2_new))
        return nothing
    end
    return g!
end

end
