# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

module ChabocheModule

using LinearAlgebra, ForwardDiff, Tensors, NLsolve, Parameters

import ..AbstractMaterial, ..AbstractMaterialState
import ..Utilities: Symm2, Symm4, I2, isotropic_elasticity_tensor, isotropic_compliance_tensor, lame, debang
import ..integrate_material!  # for method extension

# parametrically polymorphic for any type representing ℝ
export GenericChaboche, GenericChabocheDriverState, GenericChabocheParameterState, GenericChabocheVariableState

# specialization for Float64
export Chaboche, ChabocheDriverState, ChabocheParameterState, ChabocheVariableState

@with_kw mutable struct GenericChabocheDriverState{T <: Real} <: AbstractMaterialState
    time::T = zero(T)
    strain::Symm2{T} = zero(Symm2{T})
    temperature::T = zero(T)
end

"""Parameter state for Chaboche material.

The classical viscoplastic material is a special case of this model with `C1 = C2 = 0`.

Any parameter that is a `Function` should takes a single argument, the absolute
temperature.

- `theta0`: reference temperature at which thermal expansion is considered zero
- `E`: Young's modulus [N/mm^2]
- `nu`: Poisson's ratio
- `alpha`: linear thermal expansion coefficient
- `R0`: initial yield strength
- `tvp`: viscoplastic pseudo-relaxation-time (has the units of time)
- `Kn`: drag stress (has the units of stress)
- `nn`: Norton-Bailey power law exponent
- `C1`, `D1`: parameters governing behavior of backstress X1.
  C1 has the units of stress; D1 is dimensionless.
- `C2`, `D2`: parameters governing behavior of backstress X2.
- `C3`, `D3`: parameters governing behavior of backstress X3.
- `Q`: isotropic hardening saturation state (has the units of stress)
- `b`: rate of convergence to isotropic hardening saturation (dimensionless)
"""
@with_kw struct GenericChabocheParameterState{T <: Real} <: AbstractMaterialState
    theta0::T = zero(T)
    E::Function = (theta::Real -> zero(T))
    nu::Function = (theta::Real -> zero(T))
    alpha::Function = (theta::Real -> zero(T))
    R0::Function = (theta::Real -> zero(T))
    tvp::T = zero(T)
    Kn::Function = (theta::Real -> zero(T))
    nn::Function = (theta::Real -> zero(T))
    C1::Function = (theta::Real -> zero(T))
    D1::Function = (theta::Real -> zero(T))
    C2::Function = (theta::Real -> zero(T))
    D2::Function = (theta::Real -> zero(T))
    C3::Function = (theta::Real -> zero(T))
    D3::Function = (theta::Real -> zero(T))
    Q::Function = (theta::Real -> zero(T))
    b::Function = (theta::Real -> zero(T))
end

"""Problem state for Chaboche material.

- `stress`: stress tensor
- `X1`: backstress 1
- `X2`: backstress 2
- `X3`: backstress 3
- `plastic_strain`: plastic part of strain tensor
- `cumeq`: cumulative equivalent plastic strain (scalar, ≥ 0)
- `R`: yield strength
- `jacobian`: ∂σij/∂εkl
"""
@with_kw struct GenericChabocheVariableState{T <: Real} <: AbstractMaterialState
    stress::Symm2{T} = zero(Symm2{T})
    X1::Symm2{T} = zero(Symm2{T})
    X2::Symm2{T} = zero(Symm2{T})
    X3::Symm2{T} = zero(Symm2{T})
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
    state_to_vector(sigma::U, R::T, X1::U, X2::U, X3::U) where U <: Symm2{T} where T <: Real

Adaptor for `nlsolve`. Marshal the problem state into a `Vector`.
"""
function state_to_vector(sigma::U, R::T, X1::U, X2::U, X3::U) where U <: Symm2{T} where T <: Real
    return [tovoigt(sigma); R; tovoigt(X1); tovoigt(X2); tovoigt(X3)]::Vector{T}
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
    X3::Symm2{T} = fromvoigt(Symm2{T}, @view x[20:25])
    return sigma, R, X1, X2, X3
end

"""
    create_elasticity_tensor(E, nu)

Given functions `E(theta)` and `nu(theta)`, return a function
`elasticity_tensor(theta)`.
"""
function create_elasticity_tensor(E::Function, nu::Function)
    function elasticity_tensor(theta::Real)
        lambda, mu = lame(E(theta), nu(theta))
        return isotropic_elasticity_tensor(lambda, mu)
    end
    return elasticity_tensor
end

"""
    create_compliance_tensor(E, nu)

Given functions `E(theta)` and `nu(theta)`, return a function
`compliance_tensor(theta)`.
"""
function create_compliance_tensor(E::Function, nu::Function)
    function compliance_tensor(theta::Real)
        lambda, mu = lame(E(theta), nu(theta))
        return isotropic_compliance_tensor(lambda, mu)
    end
    return compliance_tensor
end

"""
    create_bulk_modulus(E, nu)

Given functions `E(theta)` and `nu(theta)`, return a function
`bulk_modulus(theta)`.
"""
function create_bulk_modulus(E::Function, nu::Function)
    function bulk_modulus(theta::Real)
        return E(theta) / (3 * (1 - 2 * nu(theta)))
    end
    return bulk_modulus
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

    theta0 = p.theta0
    Ef = p.E
    nuf = p.nu
    alphaf = p.alpha
    R0f = p.R0
    tvp = p.tvp
    Knf = p.Kn
    nnf = p.nn

    @unpack strain, time, temperature = d
    dstrain = dd.strain
    dtime = dd.time
    dtemperature = dd.temperature
    @unpack stress, X1, X2, X3, plastic_strain, cumeq, R = v

    Df = create_elasticity_tensor(Ef, nuf)
    Cf = create_compliance_tensor(Ef, nuf)
    Bf = create_bulk_modulus(Ef, nuf)

    D = Df(temperature)  # dσ/dε, i.e. ∂σij/∂εkl
    C = Cf(temperature)
    B = Bf(temperature)
    dDdT = Symm4{T}(ForwardDiff.derivative(Df, temperature))
    dBdT = ForwardDiff.derivative(Bf, temperature)
    alpha = alphaf(temperature)
    dalphadT = ForwardDiff.derivative(alphaf, temperature)
    R0 = R0f(temperature)

    # add the elastic stress increment, get the elastic trial stress
    elastic_strain = dcontract(C, stress)
    stress += (dcontract(D, dstrain)
               + dcontract(dDdT, elastic_strain) * dtemperature
               - (alpha * B
                  + ((dalphadT * B + alpha * dBdT) * (temperature - theta0)))
               * dtemperature * I2(T))

    # resulting deviatoric plastic stress (accounting for backstresses Xm)
    seff_dev = dev(stress - X1 - X2 - X3)
    # von Mises yield function
    f = sqrt(1.5)*norm(seff_dev) - (R0 + R)  # using elastic trial problem state
    # println("$(strain) $(dstrain) $(v.stress[1,1]) $(stress[1,1]) $(f)")
    if f > 0.0
        g! = create_nonlinear_system_of_equations(material)
        x0 = state_to_vector(stress, R, X1, X2, X3)
        res = nlsolve(g!, x0; method=material.options.nlsolve_method, autodiff=:forward)  # user manual: https://github.com/JuliaNLSolvers/NLsolve.jl
        converged(res) || error("Nonlinear system of equations did not converge!")
        x = res.zero
        stress, R, X1, X2, X3 = state_from_vector(x)

        # using the new problem state
        temperature_new = temperature + dtemperature
        R0 = R0f(temperature_new)
        seff_dev = dev(stress - X1 - X2 - X3)
        f = sqrt(1.5)*norm(seff_dev) - (R0 + R)

        Kn = Knf(temperature_new)
        nn = nnf(temperature_new)
        dotp = 1 / tvp * ((f >= 0.0 ? f : 0.0)/Kn)^nn  # power law viscoplasticity (Norton-Bailey type)
        dp = dotp*dtime  # |dε_p|, using backward Euler (dotp is ∂ε_p/∂t at the end of the timestep)
        n = sqrt(1.5)*seff_dev/norm(seff_dev)  # Chaboche: a (tensorial) unit direction, s.t. 2/3 * (n : n) = 1; also n = ∂f/∂σ.

        plastic_strain += dp*n
        cumeq += dp   # cumulative equivalent plastic strain (note dp ≥ 0)

        # Compute the new Jacobian, accounting for the plastic contribution. Because
        #   x ≡ [σ R X1 X2 X3]   (vector of length 25, with tensors encoded in Voigt format)
        # we have
        #   dσ/dε = (dx/dε)[1:6,1:6]
        # for which we can compute the LHS as follows:
        #   dx/dε = dx/dr dr/dε = inv(dr/dx) dr/dε ≡ (dr/dx) \ (dr/dε)
        # where r = r(x) is the residual, given by the function g!. AD can get us dr/dx automatically,
        # the other factor we will have to supply manually.
        drdx = ForwardDiff.jacobian(debang(g!), x)  # Array{25, 25}
        drde = zeros((length(x),6))                 # Array{25, 6}
        drde[1:6, 1:6] = tovoigt(D)  # elastic Jacobian. Follows from the defn. of g!.
        D = fromvoigt(Symm4, (drdx\drde)[1:6, 1:6])
    end
    variables_new = GenericChabocheVariableState{T}(stress = stress,
                                                    X1 = X1,
                                                    X2 = X2,
                                                    X3 = X3,
                                                    R = R,
                                                    plastic_strain = plastic_strain,
                                                    cumeq = cumeq,
                                                    jacobian = D)
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

Both `F` (output) and `x` (input) are length-25 vectors containing
[sigma, R, X1, X2, X3], in that order. The tensor quantities sigma,
X1, X2, X3 are encoded in Voigt format.

The function `g!` is intended to be handed over to `nlsolve`.
"""
function create_nonlinear_system_of_equations(material::GenericChaboche{T}) where T <: Real
    p = material.parameters
    v = material.variables
    dd = material.ddrivers
    d = material.drivers

    theta0 = p.theta0
    Ef = p.E
    nuf = p.nu
    alphaf = p.alpha
    R0f = p.R0
    tvp = p.tvp
    Knf = p.Kn
    nnf = p.nn
    C1f = p.C1
    D1f = p.D1
    C2f = p.C2
    D2f = p.D2
    C3f = p.C3
    D3f = p.D3
    Qf = p.Q
    bf = p.b

    # Old problem state (i.e. the problem state at the time when this equation
    # system instance was created).
    #
    # Note this does not include the elastic trial; this is the state at the
    # end of the previous timestep.

    @unpack strain, time, temperature = d
    dstrain = dd.strain
    dtime = dd.time
    dtemperature = dd.temperature
    @unpack stress, X1, X2, X3, plastic_strain, cumeq, R = v

    Df = create_elasticity_tensor(Ef, nuf)
    Cf = create_compliance_tensor(Ef, nuf)
    Bf = create_bulk_modulus(Ef, nuf)

    # We need the old elastic strain εel for computing the elastic strain
    # increment Δεel. The elastic strain isn't stored in the model, so we
    # compute it from Hooke's law:
    #   σ = D : εel
    # so
    #   εel = C : σ,
    # where C = inv(D) (with respect to the double contraction).
    C = Cf(temperature)
    elastic_strain_old = dcontract(C, stress)

    # These are needed in the residual for the new state, so we use the
    # new temperature here.
    temperature_new = temperature + dtemperature
    D = Df(temperature_new)
    B = Bf(temperature_new)
    dDdT = Symm4{T}(ForwardDiff.derivative(Df, temperature_new))
    dBdT = ForwardDiff.derivative(Bf, temperature_new)
    alpha = alphaf(temperature_new)
    dalphadT = ForwardDiff.derivative(alphaf, temperature_new)
    R0 = R0f(temperature_new)
    Kn = Knf(temperature_new)
    nn = nnf(temperature_new)
    C1 = C1f(temperature_new)
    D1 = D1f(temperature_new)
    C2 = C2f(temperature_new)
    D2 = D2f(temperature_new)
    C3 = C3f(temperature_new)
    D3 = D3f(temperature_new)
    Q = Qf(temperature_new)
    b = bf(temperature_new)

    # To compute Δεel for the residual, recall that in an elasto(-visco-)plastic
    # model with thermal effects,
    #   Δε = Δεel + Δεpl + Δεth
    # so by rearranging,
    #   Δεel = Δε - Δεpl - Δεth
    #
    # Since the total strain increment Δε is a driver, and the (visco-)plastic
    # model gives us Δεpl, we only need to compute the thermal strain increment
    # Δεth. If the thermal behavior is isotropic, it is:
    #   Δεth = Δ(α (T - T0) I)
    #        = (α ΔT + dα/dT ΔT (T - T0)) I
    #        = (α + dα/dT (T - T0)) ΔT I
    thermal_dstrain = (alpha + dalphadT * (temperature_new - theta0)) * dtemperature * I2(T)

    # Compute the residual. F is output, x is filled by NLsolve.
    # The solution is x = x* such that g(x*) = 0.
    function g!(F::V, x::V) where V <: AbstractVector{<:Real}
        stress_new, R_new, X1_new, X2_new, X3_new = state_from_vector(x)  # tentative new values from nlsolve

        seff_dev = dev(stress_new - X1_new - X2_new - X3_new)
        f = sqrt(1.5)*norm(seff_dev) - (R0 + R_new)

        dotp = 1 / tvp * ((f >= 0.0 ? f : 0.0)/Kn)^nn
        dp = dotp*dtime
        n = sqrt(1.5)*seff_dev/norm(seff_dev)

        # The equations are written in an incremental form.
        #
        # Δσ = D : Δεel  +  dD/dT : εel ΔT
        #    - (α B + (dα/dT B + α dB/dT) (T - T0)) ΔT I    (components 1:6)
        # ΔR = b (Q - R_new) |dε_p|                         (component 7)
        # ΔX1 = ((2/3) C1 n - D1 X1_new) |dε_p|             (components 8:13)
        # ΔX2 = ((2/3) C2 n - D2 X2_new) |dε_p|             (components 14:19)
        # ΔX3 = ((2/3) C3 n - D3 X3_new) |dε_p|             (components 20:25)
        #
        # where
        #
        # Δ(...) = (...)_new - (...)_old
        #
        # Note ΔT is one of the drivers, `dtemperature`.
        # Then move the terms on the RHS to the LHS to get the standard form, (stuff) = 0.
        # Also, below we avoid the multiplication and division that cancel each other
        # in the last terms of the equations for ΔX1 and ΔX2.
        #
        plastic_dstrain = dp*n
        elastic_dstrain = dstrain - plastic_dstrain - thermal_dstrain
        elastic_strain = elastic_strain_old + elastic_dstrain
        tovoigt!(view(F, 1:6),
                 stress_new - stress
                 - dcontract(D, elastic_dstrain)
                 - dcontract(dDdT, elastic_strain) * dtemperature
                 + (alpha * B
                    + ((dalphadT * B + alpha * dBdT) * (temperature - theta0)))
                 * dtemperature * I2(T))

        F[7] = R_new - R - b*(Q - R_new)*dp

        # ΔXj = ((2/3) Cj n - Dj Xj)*dp
        tovoigt!(view(F,  8:13), X1_new - X1 - dp*(2.0/3.0*C1*n - D1*X1_new))
        tovoigt!(view(F, 14:19), X2_new - X2 - dp*(2.0/3.0*C2*n - D2*X2_new))
        tovoigt!(view(F, 20:25), X3_new - X3 - dp*(2.0/3.0*C3*n - D3*X3_new))
        return nothing
    end
    return g!
end

end
