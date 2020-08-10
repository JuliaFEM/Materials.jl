# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

module DSAModule

using LinearAlgebra, ForwardDiff, Tensors, NLsolve, Parameters

import ..AbstractMaterial, ..AbstractMaterialState
import ..Utilities: Symm2, Symm4, isotropic_elasticity_tensor, lame, debang
import ..integrate_material!  # for method extension

# parametrically polymorphic for any type representing ℝ
export GenericDSA, GenericDSADriverState, GenericDSAParameterState, GenericDSAVariableState

# specialization for Float64
export DSA, DSADriverState, DSAParameterState, DSAVariableState

@with_kw mutable struct GenericDSADriverState{T <: Real} <: AbstractMaterialState
    time::T = zero(T)
    strain::Symm2{T} = zero(Symm2{T})
end

# TODO: complete this docstring
"""Parameter state for DSA (dynamic strain aging) material.

This is similar to the Chaboche model, but with additional static recovery terms.

`E`: Young's modulus
`nu`: Poisson's ratio
`R0`: initial yield strength
`Kn`: plasticity multiplier divisor (drag stress)
`nn`: plasticity multiplier exponent
`C1`, `D1`: parameters governing behavior of backstress X1
`C2`, `D2`: parameters governing behavior of backstress X2
`Q`: shift parameter for yield strength evolution
`b`: multiplier for yield strength evolution
`w`: ???
`P1`, `P2`: ???
`m`: ???
`m1`, `m2`: ???
`M1`, `M2`: ???
`ba`: ???
`xi`: ???
"""
@with_kw struct GenericDSAParameterState{T <: Real} <: AbstractMaterialState
    E::T = 0.0
    nu::T = 0.0
    R0::T = 0.0
    Kn::T = 0.0
    nn::T = 0.0
    C1::T = 0.0
    D1::T = 0.0
    C2::T = 0.0
    D2::T = 0.0
    Q::T = 0.0
    b::T = 0.0
    w::T = 0.0
    P1::T = 0.0
    P2::T = 0.0
    m::T = 0.0
    m1::T = 0.0
    m2::T = 0.0
    M1::T = 0.0
    M2::T = 0.0
    ba::T = 0.0
    xi::T = 0.0
end

# TODO: complete this docstring
"""Problem state for Chaboche material.

`stress`: stress tensor
`X1`: backstress 1
`X2`: backstress 2
`plastic_strain`: plastic part of strain tensor
`cumeq`: cumulative equivalent plastic strain (scalar, ≥ 0)
`R`: yield strength
`jacobian`: ∂σij/∂εkl
`ta`: ???
`Ra`: ???
"""
@with_kw struct GenericDSAVariableState{T <: Real} <: AbstractMaterialState
    stress::Symm2{T} = zero(Symm2{T})
    X1::Symm2{T} = zero(Symm2{T})
    X2::Symm2{T} = zero(Symm2{T})
    plastic_strain::Symm2{T} = zero(Symm2{T})
    cumeq::T = zero(T)
    R::T = zero(T)
    jacobian::Symm4{T} = zero(Symm4{T})
    ta::T = zero(T)
    Ra::T = zero(T)
end

# TODO: Does this eventually need a {T}?
@with_kw struct DSAOptions <: AbstractMaterialState
    nlsolve_method::Symbol = :trust_region
end

@with_kw mutable struct GenericDSA{T <: Real} <: AbstractMaterial
    drivers::GenericDSADriverState{T} = GenericDSADriverState{T}()
    ddrivers::GenericDSADriverState{T} = GenericDSADriverState{T}()
    variables::GenericDSAVariableState{T} = GenericDSAVariableState{T}()
    variables_new::GenericDSAVariableState{T} = GenericDSAVariableState{T}()
    parameters::GenericDSAParameterState{T} = GenericDSAParameterState{T}()
    dparameters::GenericDSAParameterState{T} = GenericDSAParameterState{T}()
    options::DSAOptions = DSAOptions()
end

DSADriverState = GenericDSADriverState{Float64}
DSAParameterState = GenericDSAParameterState{Float64}
DSAVariableState = GenericDSAVariableState{Float64}
DSA = GenericDSA{Float64}

"""
    state_to_vector(sigma::U, R::T, X1::U, X2::U, ta::T, Ra::T) where U <: Symm2{T} where T <: Real

Adaptor for `nlsolve`. Marshal the problem state into a `Vector`.
"""
function state_to_vector(sigma::U, R::T, X1::U, X2::U, ta::T, Ra::T) where U <: Symm2{T} where T <: Real
    return [tovoigt(sigma); R; tovoigt(X1); tovoigt(X2); ta; Ra]::Vector{T}
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
    ta::T = x[20]
    Ra::T = x[21]
    return sigma, R, X1, X2, ta, Ra
end

"""
    integrate_material!(material::GenericDSA{T}) where T <: Real

Material model with dynamic strain aging (DSA).

This is similar to the Chaboche material with two backstresses, with both
kinematic and isotropic hardening, but this model also features static recovery
terms.
"""
function integrate_material!(material::GenericDSA{T}) where T <: Real
    p  = material.parameters
    v  = material.variables
    dd = material.ddrivers
    d  = material.drivers
    @unpack E, nu, R0, Kn, nn, C1, D1, C2, D2, Q, b, w, P1, P2, m, m1, m2, M1, M2, ba, xi = p
    lambda, mu = lame(E, nu)

    @unpack strain, time = d
    dstrain  = dd.strain
    dtime    = dd.time
    @unpack stress, X1, X2, plastic_strain, cumeq, R, jacobian, ta, Ra = v

    # elastic part
    jacobian = isotropic_elasticity_tensor(lambda, mu)
    stress  += dcontract(jacobian, dstrain)

    # resulting deviatoric plastic stress (accounting for backstresses Xm)
    seff_dev = dev(stress - X1 - X2)
    # von Mises yield function
    f = sqrt(1.5)*norm(seff_dev) - (R0 + R + (1 - xi) * Ra)  # using elastic trial problem state
    if f > 0.0
        g! = create_nonlinear_system_of_equations(material)
        x0 = state_to_vector(stress, R, X1, X2, ta + dtime, Ra)
        res = nlsolve(g!, x0; method=material.options.nlsolve_method, autodiff = :forward)
        converged(res) || error("Nonlinear system of equations did not converge!")
        x = res.zero
        stress, R, X1, X2, ta, Ra = state_from_vector(x)

        # using the new problem state
        seff_dev = dev(stress - X1 - X2)
        f = sqrt(1.5)*norm(seff_dev) - (R0 + R + (1 - xi) * Ra)

        dotp = ((f >= 0.0 ? f : 0.0) / (Kn + xi * Ra))^nn
        dp = dotp*dtime
        n = sqrt(1.5)*seff_dev/norm(seff_dev)

        plastic_strain += dp*n
        cumeq += dp

        # Compute the new Jacobian, accounting for the plastic contribution.
        drdx = ForwardDiff.jacobian(debang(g!), x)
        drde = zeros((length(x), 6))
        drde[1:6, 1:6] = -tovoigt(jacobian)  # elastic Jacobian. Follows from the defn. of g!.
        jacobian = fromvoigt(Symm4, (drdx\drde)[1:6, 1:6])
    else
        ta += dtime
    end
    variables_new = GenericDSAVariableState{T}(stress = stress,
                                               X1 = X1,
                                               X2 = X2,
                                               R = R,
                                               plastic_strain = plastic_strain,
                                               cumeq = cumeq,
                                               jacobian = jacobian,
                                               ta = ta,
                                               Ra = Ra)
    material.variables_new = variables_new
    return nothing
end

"""
    create_nonlinear_system_of_equations(material::GenericDSA{T}) where T <: Real

Create and return an instance of the equation system for the incremental form of
the evolution equations of the DSA material.

Used internally for computing the plastic contribution in `integrate_material!`.

The input `material` represents the problem state at the end of the previous
timestep. The created equation system will hold its own copy of that state.

The equation system is represented as a mutating function `g!` that computes the
residual:

```julia
    g!(F::V, x::V) where V <: AbstractVector{<:Real}
```

Both `F` (output) and `x` (input) are length-21 vectors containing
[sigma, R, X1, X2, ta, Ra], in that order. The tensor quantities
sigma, X1, X2 are encoded in Voigt format.

The function `g!` is intended to be handed over to `nlsolve`.
"""
function create_nonlinear_system_of_equations(material::GenericDSA{T}) where T <: Real
    p  = material.parameters
    v  = material.variables
    dd = material.ddrivers
    d  = material.drivers
    @unpack E, nu, R0, Kn, nn, C1, D1, C2, D2, Q, b, w, P1, P2, m, m1, m2, M1, M2, ba, xi = p
    lambda, mu = lame(E, nu)

    # Old problem state (i.e. the problem state at the time when this equation
    # system instance was created).
    #
    # Note this does not include the elastic trial; this is the state at the
    # end of the previous timestep.
    @unpack strain, time = d
    dstrain = dd.strain
    dtime = dd.time
    @unpack stress, X1, X2, plastic_strain, cumeq, R, ta, Ra = v

    jacobian = isotropic_elasticity_tensor(lambda, mu)

    # Compute the residual. F is output, x is filled by NLsolve.
    # The solution is x = x* such that g(x*) = 0.
    function g!(F::V, x::V) where V <: AbstractVector{<:Real}
        stress_new, R_new, X1_new, X2_new, ta_new, Ra_new = state_from_vector(x)  # tentative new values from nlsolve

        seff_dev = dev(stress_new - X1_new - X2_new)
        f = sqrt(1.5)*norm(seff_dev) - (R0 + R_new + (1 - xi) * Ra_new)

        dotp = ((f >= 0.0 ? f : 0.0) / (Kn + xi * Ra_new))^nn
        dp = dotp*dtime
        n = sqrt(1.5)*seff_dev/norm(seff_dev)

        # The equations are written in an incremental form.
        # TODO: multiply the equations by -1 to make them easier to understand in the context of the rest of the model.

        dstrain_plastic = dp*n
        dstrain_elastic = dstrain - dstrain_plastic
        tovoigt!(view(F, 1:6), stress - stress_new + dcontract(jacobian, dstrain_elastic))

        F[7] = R - R_new + b*(Q - R_new)*dp

        # HACK: The zero special case is needed here to make ForwardDiff happy.
        #
        # Otherwise, when ndX1_new = 0, the components 2:end of the automatic
        # derivative of g! will be NaN, which causes the calculation of the
        # material jacobian to silently fail. This usually manifests itself as a
        # mysterious convergence failure, when this model is used in the strain
        # optimizer.
        ndX1_new = norm(dev(X1_new))
        if iszero(ndX1_new)
            JX1_new = 0.0
        else
            JX1_new = sqrt(1.5) * ndX1_new
        end
        sr1_new = (JX1_new^(m1 - 1) * X1_new) / (M1^m1)  # static recovery term
        tovoigt!(view(F, 8:13), X1 - X1_new + dp*(2.0/3.0*C1*n - D1*X1_new) - dtime*sr1_new)

        ndX2_new = norm(dev(X2_new))
        if iszero(ndX2_new)
            JX2_new = 0.0
        else
            JX2_new = sqrt(1.5) * ndX2_new
        end
        sr2_new = (JX2_new^(m2 - 1) * X2_new) / (M2^m2)  # static recovery term
        tovoigt!(view(F, 14:19), X2 - X2_new + dp*(2.0/3.0*C2*n - D2*X2_new) - dtime*sr2_new)

        Ras = P1 * (1.0 - exp(-P2 * ta_new^m))
        F[20] = ta - ta_new + dtime - (ta_new / w)*dp
        F[21] = Ra - Ra_new + ba*(Ras - Ra_new)*dp
        return nothing
    end
    return g!
end

end
