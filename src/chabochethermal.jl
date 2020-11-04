# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

module ChabocheThermalModule

using LinearAlgebra, ForwardDiff, Tensors, NLsolve, Parameters

import ..AbstractMaterial, ..AbstractMaterialState
import ..Utilities: Symm2, Symm4, isotropic_elasticity_tensor, isotropic_compliance_tensor, lame, debang
import ..integrate_material!  # for method extension

# parametrically polymorphic for any type representing ℝ
export GenericChabocheThermal, GenericChabocheThermalDriverState, GenericChabocheThermalParameterState, GenericChabocheThermalVariableState

# specialization for Float64
export ChabocheThermal, ChabocheThermalDriverState, ChabocheThermalParameterState, ChabocheThermalVariableState

"""Rank-2 identity tensor in three spatial dimensions."""
I2 = Symm2(I(3))

@with_kw mutable struct GenericChabocheThermalDriverState{T <: Real} <: AbstractMaterialState
    time::T = zero(T)
    strain::Symm2{T} = zero(Symm2{T})
    temperature::T = zero(T)
end

"""Parameter state for ChabocheThermal material.

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
@with_kw struct GenericChabocheThermalParameterState{T <: Real} <: AbstractMaterialState
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

"""Problem state for ChabocheThermal material.

- `stress`: stress tensor
- `X1`: backstress 1
- `X2`: backstress 2
- `X3`: backstress 3
- `plastic_strain`: plastic part of strain tensor
- `cumeq`: cumulative equivalent plastic strain (scalar, ≥ 0)
- `R`: yield strength
- `jacobian`: ∂σij/∂εkl
"""
@with_kw struct GenericChabocheThermalVariableState{T <: Real} <: AbstractMaterialState
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
@with_kw struct ChabocheThermalOptions <: AbstractMaterialState
    nlsolve_method::Symbol = :trust_region
end

@with_kw mutable struct GenericChabocheThermal{T <: Real} <: AbstractMaterial
    drivers::GenericChabocheThermalDriverState{T} = GenericChabocheThermalDriverState{T}()
    ddrivers::GenericChabocheThermalDriverState{T} = GenericChabocheThermalDriverState{T}()
    variables::GenericChabocheThermalVariableState{T} = GenericChabocheThermalVariableState{T}()
    variables_new::GenericChabocheThermalVariableState{T} = GenericChabocheThermalVariableState{T}()
    parameters::GenericChabocheThermalParameterState{T} = GenericChabocheThermalParameterState{T}()
    dparameters::GenericChabocheThermalParameterState{T} = GenericChabocheThermalParameterState{T}()
    options::ChabocheThermalOptions = ChabocheThermalOptions()
end

ChabocheThermalDriverState = GenericChabocheThermalDriverState{Float64}
ChabocheThermalParameterState = GenericChabocheThermalParameterState{Float64}
ChabocheThermalVariableState = GenericChabocheThermalVariableState{Float64}
ChabocheThermal = GenericChabocheThermal{Float64}

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
    elasticity_tensor(E::Function, nu::Function, theta::Real)

Usage example:

    E(θ) = ...
    ν(θ) = ...
    D(θ) = elasticity_tensor(E, ν, θ)
    dDdθ(θ) = Symm4(ForwardDiff.derivative(D, θ))
"""
function elasticity_tensor(E::Function, nu::Function, theta::Real)
    lambda, mu = lame(E(theta), nu(theta))
    return isotropic_elasticity_tensor(lambda, mu)
end

"""
    compliance_tensor(E::Function, nu::Function, theta::Real)

Usage example:

    E(θ) = ...
    ν(θ) = ...
    C(θ) = compliance_tensor(E, ν, θ)
    dCdθ(θ) = Symm4(ForwardDiff.derivative(C, θ))
"""
function compliance_tensor(E::Function, nu::Function, theta::Real)
    lambda, mu = lame(E(theta), nu(theta))
    return isotropic_compliance_tensor(lambda, mu)
end

"""
    thermal_strain_tensor(alpha::Function, theta0::Real, theta::Real)

Return the isotropic thermal strain tensor:

    εth = α(θ) (θ - θ₀) I

Here `alpha` is the linear thermal expansion coefficient, and `theta0`
is a reference temperature, at which thermal expansion is considered zero.

Usage example:

    α(θ) = ...
    θ₀ = ...
    εth(θ) = thermal_strain_tensor(α, θ₀, θ)
    dεthdθ(θ) = Symm2(ForwardDiff.derivative(εth, θ))

Given θ and Δθ, you can easily get the increment Δεth:

    Δεth(θ, Δθ) = dεthdθ(θ) * Δθ
"""
function thermal_strain_tensor(alpha::Function, theta0::Real, theta::Real)
    return alpha(theta) * (theta - theta0) * I2
end

# TODO: think about the general API
# We have `VariableState` and `ParameterState` as the parameters
# specifically so that this generalizes to other models.
# Maybe we should also have `DriverState` instead of `theta`?
#
# We should be careful to accept also `ForwardDiff.Dual`, because this stuff
# gets differentiated when computing the jacobian of the residual.
# For `yield_jacobian`, this leads to nested uses of `ForwardDiff`.
"""
    yield_criterion(state::GenericChabocheThermalVariableState{<:Real},
                    theta::T where T <: Real,
                    parameters::GenericChabocheThermalParameterState{<:Real})

Temperature-dependent yield criterion. This particular one is the von Mises
criterion for a Chaboche model with thermal effects, three backstresses,
and isotropic hardening.

`x` is a problem state, encoded into a vector. See `state_to_vector`.

`theta` is the temperature to evaluate at.

`material` is used for extracting any material parameters that
may be needed by the yield criterion. (This one needs `R0`.)

The return value is a scalar, the value of the yield function `f`.
"""
function yield_criterion(state::GenericChabocheThermalVariableState{<:Real},
                         theta::T where T <: Real,
                         parameters::GenericChabocheThermalParameterState{<:Real})
    @unpack stress, R, X1, X2, X3 = state
    @unpack R0 = parameters
    # deviatoric part of stress, accounting for plastic backstresses Xm.
    seff_dev = dev(stress - X1 - X2 - X3)
    f = sqrt(1.5)*norm(seff_dev) - (R0(theta) + R)
    return f
end

"""The jacobian of the yield criterion with respect to `state`.

The return value is a tuple as in `state_from_vector`, but each term
represents ∂f/∂V, where V is a state variable (σ, R, X1, X2, X3).

Note ∂f/∂σ = n, the normal of the yield surface.
"""
function yield_jacobian(state::GenericChabocheThermalVariableState{<:Real},
                        theta::T where T <: Real,
                        parameters::GenericChabocheThermalParameterState{<:Real})
    function f(x::AbstractVector{<:Real})
        sigma, R, X1, X2, X3 = state_from_vector(x)
        state = GenericChabocheThermalVariableState{typeof(R)}(stress=sigma,
                                                               X1=X1,
                                                               X2=X2,
                                                               X3=X3,
                                                               R=R)
        return [yield_criterion(state, theta, parameters)]::Vector
    end
    @unpack stress, R, X1, X2, X3 = state
    J = ForwardDiff.jacobian(f, state_to_vector(stress, R, X1, X2, X3))
    # Since the yield function is scalar-valued, we can pun on `state_from_vector`
    # to interpret the components of the jacobian for us.
    # The result is a row vector, so drop the singleton dimension.
    return state_from_vector(J[1,:])
end


"""
    integrate_material!(material::GenericChabocheThermal{T}) where T <: Real

ChabocheThermal material with thermal effects and three backstresses.
Both kinematic and isotropic hardening.

Viscoplastic response obeys a Norton-Bailey power law.

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
function integrate_material!(material::GenericChabocheThermal{T}) where T <: Real
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

    temperature = d.temperature
    dstrain = dd.strain
    dtime = dd.time
    dtemperature = dd.temperature
    @unpack stress, X1, X2, X3, plastic_strain, cumeq, R = v

    ff(sigma, R, X1, X2, X3, theta) = yield_criterion(GenericChabocheThermalVariableState{T}(stress=sigma,
                                                                                     X1=X1,
                                                                                     X2=X2,
                                                                                     X3=X3,
                                                                                     R=R),
                                              theta, p)
    # n = ∂f/∂σ
    nf(sigma, R, X1, X2, X3, theta) = yield_jacobian(GenericChabocheThermalVariableState{T}(stress=sigma,
                                                                                    X1=X1,
                                                                                    X2=X2,
                                                                                    X3=X3,
                                                                                    R=R),
                                             theta, p)[1]

    # Compute the elastic trial stress.
    #
    # We compute the elastic trial stress increment by using data from the
    # start of the timestep, so we have essentially a forward Euler predictor.
    #
    # Relevant equations (thermoelasto(-visco-)plastic model):
    #
    #   ε = εel + εpl + εth
    #   σ = D : εel           (Hooke's law)
    #
    # where D = D(θ) is the elastic stiffness tensor (symmetric, rank-4),
    # and θ is the absolute temperature (scalar, θ > 0).
    #
    # Thus:
    #
    #   Δσ = Δ(D : εel)
    #      = ΔD : εel + D : Δεel
    #      = (dD/dθ Δθ) : εel + D : Δεel
    #      = dD/dθ : εel Δθ + D : Δεel
    #
    # where the elastic strain increment
    #
    #   Δεel = Δε - Δεpl - Δεth
    #
    # In the elastic trial step, we temporarily assume Δεpl = 0, so then:
    #
    #   Δεel = Δε - Δεth
    #
    # The elastic stiffness tensor D is explicitly known. Its derivative dD/dθ
    # we can obtain by autodiff. The temperature increment Δθ is a driver.
    #
    # What remains to consider are the various strains. Because we store the total
    # stress σ, we can obtain the elastic strain εel by inverting Hooke's law:
    #
    #   εel = C : σ
    #
    # where C = C(θ) is the elastic compliance tensor (i.e. the inverse of
    # the elastic stiffness tensor D with respect to the double contraction),
    # and σ is a known total stress. (We have it at the start of the timestep.)
    #
    # The total strain increment Δε is a driver. So we only need to obtain Δεth.
    # The thermal strain εth is, generally speaking,
    #
    #   εth = α(θ) (θ - θ₀)
    #
    # where α is the linear thermal expansion tensor (symmetric, rank-2), and
    # θ₀ is a reference temperature, where thermal expansion is considered zero.
    #
    # We can autodiff this to obtain dεth/dθ. Then the thermal strain increment
    # Δεth is just:
    #
    #   Δεth = dεth/dθ Δθ
    #
    Cf(theta) = compliance_tensor(Ef, nuf, theta)
    C = Cf(temperature)
    elastic_strain = dcontract(C, stress)

    thermal_strainf(theta) = thermal_strain_tensor(alphaf, theta0, theta)
    thermal_strain_derivative(theta) = Symm2{T}(ForwardDiff.derivative(thermal_strainf, theta))
    thermal_dstrain = thermal_strain_derivative(temperature) * dtemperature

    Df(theta) = elasticity_tensor(Ef, nuf, theta)  # dσ/dε, i.e. ∂σij/∂εkl
    dDdthetaf(theta) = Symm4{T}(ForwardDiff.derivative(Df, theta))
    D = Df(temperature)
    dDdtheta = dDdthetaf(temperature)
    trial_elastic_dstrain = dstrain - thermal_dstrain

    stress += (dcontract(D, trial_elastic_dstrain)
               + dcontract(dDdtheta, elastic_strain) * dtemperature)

    # using elastic trial problem state
    f = ff(stress, R, X1, X2, X3, temperature)
    if f > 0.0
        g! = create_nonlinear_system_of_equations(material)
        x0 = state_to_vector(stress, R, X1, X2, X3)
        res = nlsolve(g!, x0; method=material.options.nlsolve_method, autodiff=:forward)  # user manual: https://github.com/JuliaNLSolvers/NLsolve.jl
        converged(res) || error("Nonlinear system of equations did not converge!")
        x = res.zero
        stress, R, X1, X2, X3 = state_from_vector(x)

        # using the new problem state
        temperature_new = temperature + dtemperature
        f = ff(stress, R, X1, X2, X3, temperature_new)
        n = nf(stress, R, X1, X2, X3, temperature_new)

        # Compute |∂εpl/∂t|
        Kn = Knf(temperature_new)
        nn = nnf(temperature_new)
        dotp = 1 / tvp * ((f >= 0.0 ? f : 0.0)/Kn)^nn  # power law viscoplasticity (Norton-Bailey type)
        dp = dotp*dtime  # Δp, using backward Euler (dotp is |∂εpl/∂t| at the end of the timestep)

        plastic_strain += dp*n
        cumeq += dp   # cumulative equivalent plastic strain (note Δp ≥ 0)

        # Compute the new Jacobian, accounting for the plastic contribution. Because
        #   x ≡ [σ R X1 X2 X3]   (vector of length 25, with tensors encoded in Voigt format)
        # we have
        #   dσ/dε = (dx/dε)[1:6,1:6]
        # for which we can compute the RHS as follows:
        #   dx/dε = dx/dr dr/dε = inv(dr/dx) dr/dε ≡ (dr/dx) \ (dr/dε)
        # where r = r(x) is the residual, given by the function g!. AD can get us dr/dx automatically,
        # the other factor we will have to supply manually.
        drdx = ForwardDiff.jacobian(debang(g!), x)  # Array{25, 25}
        # TODO: Update drde for this model to get quadratic convergence in Newton-Raphson (in FEM solvers).
        # TODO: Could arrange things a bit differently and use autodiff for that, too.
        drde = zeros((length(x),6))                 # Array{25, 6}
        drde[1:6, 1:6] = tovoigt(D)  # elastic Jacobian. Follows from the defn. of g!.
        D = fromvoigt(Symm4, (drdx\drde)[1:6, 1:6])
    end
    variables_new = GenericChabocheThermalVariableState{T}(stress = stress,
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
    create_nonlinear_system_of_equations(material::GenericChabocheThermal{T}) where T <: Real

Create and return an instance of the equation system for the incremental form of
the evolution equations of the ChabocheThermal material.

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
function create_nonlinear_system_of_equations(material::GenericChabocheThermal{T}) where T <: Real
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

    ff(sigma, R, X1, X2, X3, theta) = yield_criterion(GenericChabocheThermalVariableState{typeof(R)}(stress=sigma,
                                                                                             X1=X1,
                                                                                             X2=X2,
                                                                                             X3=X3,
                                                                                             R=R),
                                              theta, p)
    # n = ∂f/∂σ
    nf(sigma, R, X1, X2, X3, theta) = yield_jacobian(GenericChabocheThermalVariableState{typeof(R)}(stress=sigma,
                                                                                            X1=X1,
                                                                                            X2=X2,
                                                                                            X3=X3,
                                                                                            R=R),
                                             theta, p)[1]

    # Old problem state (i.e. the problem state at the time when this equation
    # system instance was created).
    #
    # Note this does not include the elastic trial; this is the actual state
    # at the end of the previous timestep.

    temperature = d.temperature
    dstrain = dd.strain
    dtime = dd.time
    dtemperature = dd.temperature
    @unpack stress, X1, X2, X3, plastic_strain, cumeq, R = v

    # To compute Δσ (and thus σ_new) in the residual, we need the new elastic
    # strain εel_new, as well as the elastic strain increment Δεel. By the
    # definition of Δεel,
    #
    #   εel_new = εel_old + Δεel
    #
    # The elastic strain isn't stored in the model, but the total stress is,
    # so we can obtain εel_old from Hooke's law, using the old problem state.
    #
    # The other quantity we need is Δεel. Recall that, in general:
    #
    #   Δεel = Δε - Δεpl - Δεth
    #
    # The total strain increment Δε is a driver. The (visco-)plastic model gives
    # us Δεpl (iteratively). The thermal contribution Δεth we can obtain as before.
    #
    # Thus we obtain εel and Δεel, which we can use to compute the residual for
    # the new total stress σ_new.
    #
    Cf(theta) = compliance_tensor(Ef, nuf, theta)
    C = Cf(temperature)
    elastic_strain_old = dcontract(C, stress)

    # All other quantities are needed in the computation of the residual,
    # at the new problem state, so we use the new temperature. It is a
    # driver so we have its final value without iteration.
    temperature_new = temperature + dtemperature

    thermal_strainf(theta) = thermal_strain_tensor(alphaf, theta0, theta)
    thermal_strain_derivative(theta) = Symm2{T}(ForwardDiff.derivative(thermal_strainf, theta))
    thermal_dstrain = thermal_strain_derivative(temperature_new) * dtemperature

    Df(theta) = elasticity_tensor(Ef, nuf, theta)  # dσ/dε, i.e. ∂σij/∂εkl
    dDdthetaf(theta) = Symm4{T}(ForwardDiff.derivative(Df, theta))
    D = Df(temperature_new)
    dDdtheta = dDdthetaf(temperature_new)

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

    # The evolution equations are written in an incremental form:
    #
    #   Δσ = D : Δεel + dD/dθ : εel Δθ      (components 1:6)
    #   ΔR = b (Q - R_new) Δp               (component 7)
    #   ΔXj = ((2/3) Cj n - Dj Xj_new) Δp   (components 8:13, 14:19, 20:25) (no sum)
    #
    # where
    #
    #   Δ(...) = (...)_new - (...)_old
    #
    # (Δp and n are described below.)
    #
    # Then in each equation, move the terms on the RHS to the LHS to get
    # the standard form, (stuff) = 0. Then the LHS is the residual.
    #
    # The viscoplastic response is updated by:
    #
    #   Δεpl = Δp n
    #
    # where
    #
    #   Δp = p' Δt
    #   p' = 1/tvp * (<f> / Kn)^nn  (Norton-Bailey power law; <...>: Macaulay brackets)
    #   f = √(3/2 dev(σ_eff) : dev(σ_eff)) - (R0 + R)
    #   σ_eff = σ - ∑ Xj
    #   n = ∂f/∂σ
    #
    # Compute the residual. F is output, x is filled by NLsolve.
    # The solution is x = x* such that g(x*) = 0.
    function g!(F::V, x::V) where V <: AbstractVector{<:Real}
        stress_new, R_new, X1_new, X2_new, X3_new = state_from_vector(x)  # tentative new values from nlsolve

        f = ff(stress_new, R_new, X1_new, X2_new, X3_new, temperature_new)
        n = nf(stress_new, R_new, X1_new, X2_new, X3_new, temperature_new)

        dotp = 1 / tvp * ((f >= 0.0 ? f : 0.0)/Kn)^nn
        dp = dotp*dtime

        plastic_dstrain = dp*n
        elastic_dstrain = dstrain - plastic_dstrain - thermal_dstrain
        elastic_strain = elastic_strain_old + elastic_dstrain
        tovoigt!(view(F, 1:6),
                 stress_new - stress
                 - dcontract(D, elastic_dstrain)
                 - dcontract(dDdtheta, elastic_strain) * dtemperature)

        F[7] = R_new - R - b*(Q - R_new)*dp

        tovoigt!(view(F,  8:13), X1_new - X1 - dp*(2.0/3.0*C1*n - D1*X1_new))
        tovoigt!(view(F, 14:19), X2_new - X2 - dp*(2.0/3.0*C2*n - D2*X2_new))
        tovoigt!(view(F, 20:25), X3_new - X3 - dp*(2.0/3.0*C3*n - D3*X3_new))
        return nothing
    end
    return g!
end

end
