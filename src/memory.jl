# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

module MemoryModule

using LinearAlgebra, ForwardDiff, Tensors, NLsolve, Parameters

import ..AbstractMaterial, ..AbstractMaterialState
import ..Utilities: Symm2, Symm4, isotropic_elasticity_tensor, lame, debang
import ..integrate_material!  # for method extension

# parametrically polymorphic for any type representing ℝ
export GenericMemory, GenericMemoryDriverState, GenericMemoryParameterState, GenericMemoryVariableState

# specialization for Float64
export Memory, MemoryDriverState, MemoryParameterState, MemoryVariableState

@with_kw mutable struct GenericMemoryDriverState{T <: Real} <: AbstractMaterialState
    time::T = zero(T)
    strain::Symm2{T} = zero(Symm2{T})
end

"""Parameter state for Memory material.

- `E`: Young's modulus
- `nu`: Poisson's ratio
- `R0`: initial yield strength
- `Kn`: plasticity multiplier divisor (drag stress)
- `nn`: plasticity multiplier exponent
- `C1`, `D1`: parameters governing behavior of backstress X1
- `C2`, `D2`: parameters governing behavior of backstress X2
- `Q0`: The initial isotropic hardening saturation value. Has the units of stress.
- `QM`: The asymptotic isotropic hardening saturation value reached with high strain amplitude.

  Has the units of stress.

- `mu`: Controls the rate of evolution of the strain-amplitude dependent isotropic hardening saturation value.
- `b`: Controls the rate of evolution for isotropic hardening.
- `eta`: Controls the balance between memory surface kinematic and isotropic hardening.

  Dimensionless, support `[0,1]`.
  At `0`, the memory surface hardens kinematically.
  At `1`, the memory surface hardens isotropically.

  Initially, the value `1/2` was used by several authors. Later, values `< 1/2` have been suggested
  to capture the progressive process of memorization.

- `m`: memory evanescence exponent. Controls the non-linearity of the memory evanescence.
- `pt`: threshold of equivalent plastic strain, after which the memory evanescence starts.
- `xi`: memory evanescence strength multiplier.
"""
@with_kw struct GenericMemoryParameterState{T <: Real} <: AbstractMaterialState
    E::T = 0.0
    nu::T = 0.0
    R0::T = 0.0
    Kn::T = 0.0
    nn::T = 0.0
    C1::T = 0.0
    D1::T = 0.0
    C2::T = 0.0
    D2::T = 0.0
    Q0::T = 0.0
    QM::T = 0.0
    mu::T = 0.0
    b::T = 0.0
    eta::T = 0.0
    m::T = 0.0
    pt::T = 0.0
    xi::T = 0.0
end

"""Problem state for Memory material.

- `stress`: stress tensor
- `X1`: backstress 1
- `X2`: backstress 2
- `plastic_strain`: plastic part of strain tensor
- `cumeq`: cumulative equivalent plastic strain (scalar, ≥ 0)
- `R`: yield strength
- `q`: size of the strain memory surface (~plastic strain amplitude)
- `zeta`: strain memory surface kinematic hardening variable
- `jacobian`: ∂σij/∂εkl
"""
@with_kw struct GenericMemoryVariableState{T <: Real} <: AbstractMaterialState
    stress::Symm2{T} = zero(Symm2{T})
    X1::Symm2{T} = zero(Symm2{T})
    X2::Symm2{T} = zero(Symm2{T})
    plastic_strain::Symm2{T} = zero(Symm2{T})
    cumeq::T = zero(T)
    R::T = zero(T)
    q::T = zero(T)
    zeta::Symm2{T} = zero(Symm2{T})
    jacobian::Symm4{T} = zero(Symm4{T})
end

# TODO: Does this eventually need a {T}?
@with_kw struct MemoryOptions <: AbstractMaterialState
    nlsolve_method::Symbol = :trust_region
end

@with_kw mutable struct GenericMemory{T <: Real} <: AbstractMaterial
    drivers::GenericMemoryDriverState{T} = GenericMemoryDriverState{T}()
    ddrivers::GenericMemoryDriverState{T} = GenericMemoryDriverState{T}()
    variables::GenericMemoryVariableState{T} = GenericMemoryVariableState{T}()
    variables_new::GenericMemoryVariableState{T} = GenericMemoryVariableState{T}()
    parameters::GenericMemoryParameterState{T} = GenericMemoryParameterState{T}()
    dparameters::GenericMemoryParameterState{T} = GenericMemoryParameterState{T}()
    options::MemoryOptions = MemoryOptions()
end

MemoryDriverState = GenericMemoryDriverState{Float64}
MemoryParameterState = GenericMemoryParameterState{Float64}
MemoryVariableState = GenericMemoryVariableState{Float64}
Memory = GenericMemory{Float64}

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
    strain_memory_explicit_update(q, zeta, plastic_strain, dp, cumeq, pt, n, eta, xi, m)

Internal helper function for what it says on the tin.

Return `(dq, dzeta)`, the computed increments for `q` and `zeta`
for the given input.
"""
function strain_memory_explicit_update(q, zeta, plastic_strain, dp, cumeq, pt, n, eta, xi, m)
    dq = zero(q)
    dzeta = zero(zeta)
    JF = sqrt(1.5)*norm(dev(plastic_strain - zeta))
    FF = 2.0/3.0*JF - q
    if FF > 0.0
        nF = 1.5*dev(plastic_strain - zeta)/JF
        nnF = dcontract(n, nF)
        if nnF > 0
            dq = 2.0/3.0*eta*nnF*dp
            dzeta = 2.0/3.0*(1.0 - eta)*nnF*nF*dp
        end
    else
        # Memory evanescence term
        if cumeq >= pt
            dq = -xi*q^m*dp
        end
    end
    return dq, dzeta
end

"""
    integrate_material!(material::GenericMemory{T}) where T <: Real

Material model with a strain memory effect.

This is similar to the Chaboche material with two backstresses, with both
kinematic and isotropic hardening, but this model also features a strain
memory term.

Strain memory is used to be able to model strain amplitude-dependent isotropic
hardening. In practice, the transition from a tensile test curve to cyclic
behavior can be captured with this model.

See:

    D. Nouailhas, J.-L. Chaboche, S. Savalle, G. Cailletaud. On the constitutive
    equations for cyclic plasticity under nonproportional loading. International
    Journal of Plasticity 1(4) (1985), 317--330.
    https://doi.org/10.1016/0749-6419(85)90018-X
"""
function integrate_material!(material::GenericMemory{T}) where T <: Real
    p = material.parameters
    v = material.variables
    dd = material.ddrivers
    d = material.drivers
    @unpack E, nu, R0, Kn, nn, C1, D1, C2, D2, Q0, QM, mu, b, eta, m, pt, xi = p
    lambda, elastic_mu = lame(E, nu)

    @unpack strain, time = d
    dstrain = dd.strain
    dtime = dd.time
    @unpack stress, X1, X2, plastic_strain, cumeq, R, q, zeta, jacobian = v

    # elastic part
    jacobian = isotropic_elasticity_tensor(lambda, elastic_mu)
    stress += dcontract(jacobian, dstrain)

    # resulting deviatoric plastic stress (accounting for backstresses Xm)
    seff_dev = dev(stress - X1 - X2)
    # von Mises yield function
    f = sqrt(1.5)*norm(seff_dev) - (R0 + R)  # using elastic trial problem state
    if f > 0.0
        # Explicit update to memory-surface
        g! = create_nonlinear_system_of_equations(material)
        x0 = state_to_vector(stress, R, X1, X2)
        res = nlsolve(g!, x0; method=material.options.nlsolve_method, autodiff = :forward)
        converged(res) || error("Nonlinear system of equations did not converge!")
        x = res.zero
        stress, R, X1, X2 = state_from_vector(x)

        # using the new problem state
        seff_dev = dev(stress - X1 - X2)
        f = sqrt(1.5)*norm(seff_dev) - (R0 + R)

        dotp = ((f >= 0.0 ? f : 0.0)/Kn)^nn
        dp = dotp*dtime
        n = sqrt(1.5)*seff_dev/norm(seff_dev)

        plastic_strain += dp*n
        cumeq += dp

        dq, dzeta = strain_memory_explicit_update(q, zeta, plastic_strain, dp, cumeq, pt, n, eta, xi, m)
        q += dq
        zeta += dzeta

        # Compute the new Jacobian, accounting for the plastic contribution.
        drdx = ForwardDiff.jacobian(debang(g!), x)
        drde = zeros((length(x),6))
        drde[1:6, 1:6] = -tovoigt(jacobian)
        jacobian = fromvoigt(Symm4, (drdx\drde)[1:6, 1:6])
    end
    variables_new = GenericMemoryVariableState{T}(stress = stress,
                                                  X1 = X1,
                                                  X2 = X2,
                                                  R = R,
                                                  plastic_strain = plastic_strain,
                                                  cumeq = cumeq,
                                                  q = q,
                                                  zeta = zeta,
                                                  jacobian = jacobian)
    material.variables_new = variables_new
    return nothing
end

"""
    create_nonlinear_system_of_equations(material::GenericMemory{T}) where T <: Real

Create and return an instance of the equation system for the incremental form of
the evolution equations of the Memory material.

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
function create_nonlinear_system_of_equations(material::GenericMemory{T}) where T <: Real
    p = material.parameters
    v = material.variables
    dd = material.ddrivers
    d = material.drivers
    @unpack E, nu, R0, Kn, nn, C1, D1, C2, D2, Q0, QM, mu, b, eta, m, pt, xi = p
    lambda, elastic_mu = lame(E, nu)

    # Old problem state (i.e. the problem state at the time when this equation
    # system instance was created).
    #
    # Note this does not include the elastic trial; this is the state at the
    # end of the previous timestep.
    @unpack strain, time = d
    dstrain = dd.strain
    dtime = dd.time
    @unpack stress, X1, X2, plastic_strain, cumeq, R, q, zeta, jacobian = v

    jacobian = isotropic_elasticity_tensor(lambda, elastic_mu)

    # Explicit update of memory surface.
    #
    # Compute the residual. F is output, x is filled by NLsolve.
    # The solution is x = x* such that g(x*) = 0.
    function g!(F::V, x::V) where V <: AbstractVector{<:Real}
        stress_new, R_new, X1_new, X2_new = state_from_vector(x)  # tentative new values from nlsolve

        seff_dev = dev(stress_new - X1_new - X2_new)
        f = sqrt(1.5)*norm(seff_dev) - (R0 + R_new)

        dotp = ((f >= 0.0 ? f : 0.0)/Kn)^nn
        dp = dotp*dtime
        n = sqrt(1.5)*seff_dev/norm(seff_dev)

        dstrain_plastic = dp*n

        # Strain memory - explicit update
        plastic_strain_new = plastic_strain + dstrain_plastic
        dq, dzeta = strain_memory_explicit_update(q, zeta, plastic_strain_new, dp, cumeq, pt, n, eta, xi, m)
        q_new = q + dq
        zeta_new = zeta + dzeta

        # The equations are written in an incremental form.
        # TODO: multiply the equations by -1 to make them easier to understand in the context of the rest of the model.

        dstrain_elastic = dstrain - dstrain_plastic
        tovoigt!(view(F, 1:6), stress - stress_new + dcontract(jacobian, dstrain_elastic))

        F[7] = R - R_new + b*((QM + (Q0 - QM)*exp(-2.0*mu*q_new)) - R_new)*dp

        tovoigt!(view(F, 8:13), X1 - X1_new + dp*(2.0/3.0*C1*n - D1*X1_new))
        tovoigt!(view(F, 14:19), X2 - X2_new + dp*(2.0/3.0*C2*n - D2*X2_new))
        return nothing
    end
    return g!
end

end
