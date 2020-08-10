# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

# TODO: write docstrings for all public functions

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

        # Strain memory - explicit update
        JF = sqrt(1.5)*norm(dev(plastic_strain - zeta))
        FF = 2.0/3.0*JF - q
        if FF > 0.0
            nF = 1.5*dev(plastic_strain - zeta)/JF
            nnF = dcontract(n, nF)
            if nnF > 0
                q += 2.0/3.0*eta*nnF*dp
                zeta += 2.0/3.0*(1.0 - eta)*nnF*nF*dp
            end
        else
            # Memory evanescence term
            if cumeq >= pt
                q += -xi*q^m*dp
            end
        end

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
        JF = sqrt(1.5)*norm(dev(plastic_strain_new - zeta))
        FF = 2.0/3.0*JF - q
        if FF > 0.0
            nF = 1.5*dev(plastic_strain_new - zeta)/JF
            nnF = dcontract(n, nF)
            if nnF > 0
                q_new = q + 2.0/3.0*eta*nnF*dp
                zeta_new = zeta + 2.0/3.0*(1.0 - eta)*nnF*nF*dp
            else
                q_new = q
                zeta_new = zeta
            end
        else
            # Memory evanescence term
            p_new = cumeq + dp
            if p_new > pt
                q_new = q - xi*q^m*dp
            else
                q_new = q
            end
            zeta_new = zeta
        end

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
