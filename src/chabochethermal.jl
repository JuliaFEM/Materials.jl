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

# TODO: hierarchize parameters: elasticity, kinematic hardening, isotropic hardening, ...
# plasticity: yield criterion, flow rule, hardening
"""Parameter state for ChabocheThermal material.

The classical viscoplastic material is a special case of this model with `C1 = C2 = C3 = 0`.

Maximum hardening for each backstress is `Cj / Dj`.

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
    theta0::T = zero(T)  # reference temperature for thermal behavior
    # basic material parameters
    E::Function = (theta::Real -> zero(T))
    nu::Function = (theta::Real -> zero(T))
    alpha::Function = (theta::Real -> zero(T))
    R0::Function = (theta::Real -> zero(T))
    # parameters for viscoplastic overstress model
    tvp::T = zero(T)
    Kn::Function = (theta::Real -> zero(T))
    nn::Function = (theta::Real -> zero(T))
    # kinematic hardening parameters
    C1::Function = (theta::Real -> zero(T))
    D1::Function = (theta::Real -> zero(T))
    C2::Function = (theta::Real -> zero(T))
    D2::Function = (theta::Real -> zero(T))
    C3::Function = (theta::Real -> zero(T))
    D3::Function = (theta::Real -> zero(T))
    # isotropic hardening parameters
    Q::Function = (theta::Real -> zero(T))
    b::Function = (theta::Real -> zero(T))
end

"""Problem state for ChabocheThermal material.

- `stress`: stress tensor
- `R`: yield strength (isotropic hardening)
- `X1`: backstress 1 (kinematic hardening)
- `X2`: backstress 2 (kinematic hardening)
- `X3`: backstress 3 (kinematic hardening)
- `plastic_strain`: plastic part of strain tensor
- `cumeq`: cumulative equivalent plastic strain (scalar, ≥ 0)
- `jacobian`: ∂σij/∂εkl (algorithmic)

The other `dXXXdYYY` properties are the algorithmic jacobians for the
indicated variables.

The elastic and thermal contributions to the strain tensor are not stored.
To get them:

    θ₀ = ...
    θ = ...
    p = material.parameters
    v = material.variables

    C(θ) = compliance_tensor(p.E, p.nu, θ)
    elastic_strain = dcontract(C(θ), v.stress)

    thermal_strain = thermal_strain_tensor(p.alpha, θ₀, θ)

Then it holds that:

    material.drivers.strain = elastic_strain + v.plastic_strain + thermal_strain
"""
@with_kw struct GenericChabocheThermalVariableState{T <: Real} <: AbstractMaterialState
    stress::Symm2{T} = zero(Symm2{T})
    R::T = zero(T)
    X1::Symm2{T} = zero(Symm2{T})
    X2::Symm2{T} = zero(Symm2{T})
    X3::Symm2{T} = zero(Symm2{T})
    plastic_strain::Symm2{T} = zero(Symm2{T})
    cumeq::T = zero(T)
    jacobian::Symm4{T} = zero(Symm4{T})
    dRdstrain::Symm2{T} = zero(Symm2{T})
    dX1dstrain::Symm4{T} = zero(Symm4{T})
    dX2dstrain::Symm4{T} = zero(Symm4{T})
    dX3dstrain::Symm4{T} = zero(Symm4{T})
    dstressdtemperature::Symm2{T} = zero(Symm2{T})
    dRdtemperature::T = zero(T)
    dX1dtemperature::Symm2{T} = zero(Symm2{T})
    dX2dtemperature::Symm2{T} = zero(Symm2{T})
    dX3dtemperature::Symm2{T} = zero(Symm2{T})
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
    dDdθ(θ) = gradient(D, θ)
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
    dCdθ(θ) = gradient(C, θ)
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
    dεthdθ(θ) = gradient(εth, θ)

Given θ and Δθ, you can easily get the increment Δεth:

    Δεth(θ, Δθ) = dεthdθ(θ) * Δθ
"""
function thermal_strain_tensor(alpha::Function, theta0::Real, theta::Real)
    return alpha(theta) * (theta - theta0) * I2
end

# TODO: Add this interface to the general API in `AbstractMaterial`?
#
# We should be careful to accept also `ForwardDiff.Dual`, because this stuff
# gets differentiated when computing the jacobian of the residual.
# For `yield_jacobian`, that leads to nested uses of `ForwardDiff`.
"""
    yield_criterion(state::GenericChabocheThermalVariableState{<:Real},
                    drivers::GenericChabocheThermalDriverState{<:Real},
                    parameters::GenericChabocheThermalParameterState{<:Real})

Temperature-dependent yield criterion. This particular one is the von Mises
criterion for a Chaboche model with thermal effects, three backstresses,
and isotropic hardening.

`state` should contain `stress`, `R`, `X1`, `X2`, `X3`.
`drivers` should contain `temperature`.
`parameters` should contain `R0`, a function of temperature.

Other properties of the structures are not used by this function.

The return value is a scalar, the value of the yield function `f`.
"""
function yield_criterion(state::GenericChabocheThermalVariableState{<:Real},
                         drivers::GenericChabocheThermalDriverState{<:Real},
                         parameters::GenericChabocheThermalParameterState{<:Real})
    @unpack stress, R, X1, X2, X3 = state
    @unpack temperature = drivers
    @unpack R0 = parameters
    # deviatoric part of stress, accounting for plastic backstresses Xm.
    seff_dev = dev(stress - X1 - X2 - X3)
    f = sqrt(1.5)*norm(seff_dev) - (R0(temperature) + R)
    return f
end

"""
    yield_jacobian(state::GenericChabocheThermalVariableState{<:Real},
                   drivers::GenericChabocheThermalDriverState{<:Real},
                   parameters::GenericChabocheThermalParameterState{<:Real})

Compute `n = ∂f/∂σ`.

`state` should contain `stress`, `R`, `X1`, `X2`, `X3`.
`drivers` should contain `temperature`.
`parameters` should contain `R0`, a function of temperature.

Other properties of the structures are not used by this function.

The return value is the symmetric rank-2 tensor `n`.
"""
function yield_jacobian(state::GenericChabocheThermalVariableState{<:Real},
                        drivers::GenericChabocheThermalDriverState{<:Real},
                        parameters::GenericChabocheThermalParameterState{<:Real})
    # We only need ∂f/∂σ, so let's compute only that to make this run faster.
    #
    # # TODO: The `gradient` wrapper of `Tensors.jl` is nice, but it doesn't tag its Dual.
    # #
    # # When using `Tensors.gradient` in `yield_jacobian` (n = ∂f/∂σ), the
    # # differentiation of `yield_criterion` with respect to stress doesn't work
    # # when computing the temperature jacobian for the residual function, which
    # # needs ∂n/∂θ = ∂²f/∂σ∂θ. `ForwardDiff` fails to find an ordering for the
    # # `Dual` terms (the temperature Dual having a tag, but the stress Dual not).
    # #
    # # Using `ForwardDiff.jacobian` directly, both Duals are tagged, so this works.
    #
    # @unpack stress, R, X1, X2, X3 = state
    # function f(stress::Symm2{<:Real})
    #     state = GenericChabocheThermalVariableState{eltype(stress)}(stress=stress,
    #                                                                 R=R,
    #                                                                 X1=X1,
    #                                                                 X2=X2,
    #                                                                 X3=X3)
    #     return yield_criterion(state, drivers, parameters)
    # end
    # return gradient(f, stress)
    @unpack stress, R, X1, X2, X3 = state
    marshal(tensor::Symm2) = tovoigt(tensor)
    unmarshal(x::AbstractVector{T}) where T <: Real = fromvoigt(Symm2{T}, x)
    function f(x::AbstractVector{<:Real})  # x = stress
        state = GenericChabocheThermalVariableState{eltype(x)}(stress=unmarshal(x),
                                                               X1=X1,
                                                               X2=X2,
                                                               X3=X3,
                                                               R=R)
        return [yield_criterion(state, drivers, parameters)]::Vector
    end
    J = ForwardDiff.jacobian(f, marshal(stress))
    # The result is a row vector, so drop the singleton dimension.
    return unmarshal(J[1,:])
end

"""
    overstress_function(state::GenericChabocheThermalVariableState{<:Real},
                           drivers::GenericChabocheThermalDriverState{<:Real},
                           parameters::GenericChabocheThermalParameterState{<:Real})

Norton-Bailey type power law.

`parameters` should contain `tvp`, `Kn` and `nn`.
`drivers` should contain `temperature`.

Additionally, `state`, `drivers` and `parameters` will be passed to
`yield_criterion`.

The return value is `dotp` that can be used in `dp = dotp * dtime`.
"""
function overstress_function(state::GenericChabocheThermalVariableState{<:Real},
                             drivers::GenericChabocheThermalDriverState{<:Real},
                             parameters::GenericChabocheThermalParameterState{<:Real})
    f = yield_criterion(state, drivers, parameters)
    @unpack tvp, Kn, nn = parameters
    @unpack temperature = drivers
    K = Kn(temperature)
    n = nn(temperature)
    return 1 / tvp * ((f >= 0.0 ? f : 0.0) / K)^n
end


"""
    integrate_material!(material::GenericChabocheThermal{T}) where T <: Real

Chaboche viscoplastic material with thermal effects. The model includes
kinematic hardening with three backstresses, and isotropic hardening.

Let the prime (') denote the time derivative. The evolution equations are:

  σ' = D : εel' + dD/dθ : εel θ'
  R' = b (Q - R) p'
  Xj' = ((2/3) Cj n - Dj Xj) p'   (no sum)

where j = 1, 2, 3. The strain consists of elastic, thermal and viscoplastic
contributions:

  ε = εel + εth + εpl

Outside the elastic region, the viscoplastic strain response is given by:

  εpl' = n p'

where

  n = ∂f/∂σ

and p' obeys a Norton-Bailey power law:

  p' = 1/tvp * (<f> / Kn)^nn

Here <...> are the Macaulay brackets (a.k.a. positive part), and
the yield criterion is of the von Mises type:

  f = √(3/2 dev(σ_eff) : dev(σ_eff)) - (R0 + R)
  σ_eff = σ - ∑ Xj

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

    temperature = d.temperature
    dstrain = dd.strain
    dtime = dd.time
    dtemperature = dd.temperature
    @unpack stress, X1, X2, X3, plastic_strain, cumeq, R = v

    VariableState{U} = GenericChabocheThermalVariableState{U}
    DriverState{U} = GenericChabocheThermalDriverState{U}
    ff(sigma, R, X1, X2, X3, theta) = yield_criterion(VariableState{T}(stress=sigma, R=R, X1=X1, X2=X2, X3=X3),
                                                      DriverState{T}(temperature=theta),
                                                      p)
    # n = ∂f/∂σ
    nf(sigma, R, X1, X2, X3, theta) = yield_jacobian(VariableState{T}(stress=sigma, R=R, X1=X1, X2=X2, X3=X3),
                                                     DriverState{T}(temperature=theta),
                                                     p)
    # p'  (dp = p' * dtime)
    dotpf(sigma, R, X1, X2, X3, theta) = overstress_function(VariableState{T}(stress=sigma, R=R, X1=X1, X2=X2, X3=X3),
                                                             DriverState{T}(temperature=theta),
                                                             p)

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

    # This is a function so we can autodiff it to get the algorithmic jacobian in the elastic region.
    # Δσ = D : Δεel + dD/dθ : εel Δθ
    function elastic_dstress(dstrain::Symm2{<:Real}, dtemperature::Real)
        local temperature_new = temperature + dtemperature

        thermal_strainf(theta) = thermal_strain_tensor(alphaf, theta0, theta)
        thermal_dstrain = thermal_strainf(temperature_new) - thermal_strainf(temperature)
        trial_elastic_dstrain = dstrain - thermal_dstrain

        Df(theta) = elasticity_tensor(Ef, nuf, theta)  # dσ/dε, i.e. ∂σij/∂εkl
        dDdthetaf(theta) = gradient(Df, theta)

        # Evaluating `Df` and `dDdthetaf` at `temperature_new` eliminates integrator drift
        # in cyclic uniaxial loading conditions inside the elastic region.
        # Note in the second term we use the *old* elastic strain.
        return (dcontract(Df(temperature_new), trial_elastic_dstrain)
                + dcontract(dDdthetaf(temperature_new), elastic_strain) * dtemperature)
    end

    stress += elastic_dstress(dstrain, dtemperature)

    # using elastic trial problem state
    if ff(stress, R, X1, X2, X3, temperature) > 0.0  # plastic region
        rx!, rdstrain, rtemperature = create_nonlinear_system_of_equations(material)
        x0 = state_to_vector(stress, R, X1, X2, X3)
        res = nlsolve(rx!, x0; method=material.options.nlsolve_method, autodiff=:forward)  # user manual: https://github.com/JuliaNLSolvers/NLsolve.jl
        converged(res) || error("Nonlinear system of equations did not converge!")
        x = res.zero
        stress, R, X1, X2, X3 = state_from_vector(x)

        # using the new problem state
        temperature_new = temperature + dtemperature

        # Compute the new plastic strain
        dotp = dotpf(stress, R, X1, X2, X3, temperature_new)
        n = nf(stress, R, X1, X2, X3, temperature_new)

        dp = dotp * dtime  # Δp, using backward Euler (dotp is |∂εpl/∂t| at the end of the timestep)
        plastic_strain += dp * n
        cumeq += dp   # cumulative equivalent plastic strain (note Δp ≥ 0)

        # Compute the new algorithmic jacobian Jstrain by implicit differentiation of the residual function,
        # using `ForwardDiff` to compute the derivatives. Details in `create_nonlinear_system_of_equations`.
        # We compute ∂V/∂D ∀ V ∈ state, D ∈ drivers (excluding time).
        drdx = ForwardDiff.jacobian(debang(rx!), x)

        # Here we don't bother with offdiagscale, since this Voigt conversion is just a marshaling.
        # All `rdstrain` does with the Voigt `dstrain` is to unmarshal it back into a tensor.
        # All computations are performed in tensor format.
        rdstrainf(dstrain) = rdstrain(stress, R, X1, X2, X3, dstrain)  # at solution point
        drdstrain = ForwardDiff.jacobian(rdstrainf, tovoigt(dstrain))
        Jstrain = -drdx \ drdstrain
        jacobian = fromvoigt(Symm4, Jstrain[1:6, 1:6])
        dRdstrain = fromvoigt(Symm2, Jstrain[7, 1:6])
        dX1dstrain = fromvoigt(Symm4, Jstrain[8:13, 1:6])
        dX2dstrain = fromvoigt(Symm4, Jstrain[14:19, 1:6])
        dX3dstrain = fromvoigt(Symm4, Jstrain[20:25, 1:6])

        rtemperaturef(theta) = rtemperature(stress, R, X1, X2, X3, theta)  # at solution point
        drdtemperature = ForwardDiff.jacobian(rtemperaturef, [temperature_new])
        Jtemperature = -drdx \ drdtemperature
        dstressdtemperature = fromvoigt(Symm2, Jtemperature[1:6, 1])
        dRdtemperature = Jtemperature[7, 1]
        dX1dtemperature = fromvoigt(Symm2, Jtemperature[8:13, 1])
        dX2dtemperature = fromvoigt(Symm2, Jtemperature[14:19, 1])
        dX3dtemperature = fromvoigt(Symm2, Jtemperature[20:25, 1])
    else  # elastic region
        # TODO: update R (thermal effects), see if Xs also need updating

        jacobian = gradient(((dstrain) -> elastic_dstress(dstrain, dtemperature)),
                            dstrain)
        dstressdtemperature = gradient(((dtemperature) -> elastic_dstress(dstrain, dtemperature)),
                                       dtemperature)

        # In the elastic region, the plastic variables stay constant,
        # so their jacobians vanish.
        dRdstrain = zero(Symm2{T})
        dX1dstrain = zero(Symm4{T})
        dX2dstrain = zero(Symm4{T})
        dX3dstrain = zero(Symm4{T})
        dRdtemperature = zero(T)
        dX1dtemperature = zero(Symm2{T})
        dX2dtemperature = zero(Symm2{T})
        dX3dtemperature = zero(Symm2{T})
    end
    variables_new = VariableState{T}(stress=stress,
                                     R=R,
                                     X1=X1,
                                     X2=X2,
                                     X3=X3,
                                     plastic_strain=plastic_strain,
                                     cumeq=cumeq,
                                     jacobian=jacobian,
                                     dRdstrain=dRdstrain,
                                     dX1dstrain=dX1dstrain,
                                     dX2dstrain=dX2dstrain,
                                     dX3dstrain=dX3dstrain,
                                     dstressdtemperature=dstressdtemperature,
                                     dRdtemperature=dRdtemperature,
                                     dX1dtemperature=dX1dtemperature,
                                     dX2dtemperature=dX2dtemperature,
                                     dX3dtemperature=dX3dtemperature)
    material.variables_new = variables_new
    return nothing
end

"""
    create_nonlinear_system_of_equations(material::GenericChabocheThermal{T}) where T <: Real

Create and return an instance of the equation system for the incremental form of
the evolution equations.

Used internally for computing the viscoplastic contribution in `integrate_material!`.

The input `material` represents the problem state at the end of the previous
timestep. The created equation system will hold its own copy of that state.

The equation system is represented as a mutating function `r!` that computes the
residual:

```julia
    r!(F::V, x::V) where V <: AbstractVector{<:Real}
```

Both `F` (output) and `x` (input) are length-25 vectors containing
[sigma, R, X1, X2, X3], in that order. The tensor quantities sigma,
X1, X2, X3 are encoded in Voigt format.

The function `r!` is intended to be handed over to `nlsolve`.
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
    C1f = p.C1
    D1f = p.D1
    C2f = p.C2
    D2f = p.D2
    C3f = p.C3
    D3f = p.D3
    Qf = p.Q
    bf = p.b

    VariableState{U} = GenericChabocheThermalVariableState{U}
    DriverState{U} = GenericChabocheThermalDriverState{U}
    # n = ∂f/∂σ
    nf(sigma, R, X1, X2, X3, theta) = yield_jacobian(VariableState{eltype(sigma)}(stress=sigma, R=R, X1=X1, X2=X2, X3=X3),
                                                     DriverState{typeof(theta)}(temperature=theta),
                                                     p)
    # p'  (dp = p' * dtime)
    dotpf(sigma, R, X1, X2, X3, theta) = overstress_function(VariableState{eltype(sigma)}(stress=sigma, R=R, X1=X1, X2=X2, X3=X3),
                                                             DriverState{typeof(theta)}(temperature=theta),
                                                             p)

    # Old problem state (i.e. the problem state at the time when this equation
    # system instance was created).
    #
    # Note this does not include the elastic trial; this is the actual state
    # at the end of the previous timestep.

    temperature = d.temperature
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

    # To solve the equation system, we need to parameterize the residual function
    # by all unknowns.
    #
    # To obtain the algorithmic jacobian ∂(Δσ)/∂(Δε), first keep in mind that as
    # far as the algorithm is concerned, σ_old and ε_old are constants. Therefore,
    # ∂(...)/∂(Δσ) = ∂(...)/∂(σ_new), and similarly for Δε, ε_new.
    #
    # Let r denote the residual function. For simplicity, consider only the increments
    # Δε, Δσ for now (we will generalize below). At a solution point, we have:
    #
    #   r(Δε, Δσ) = 0
    #
    # **On the condition that** we stay on the solution surface - i.e. it remains true
    # that r = 0 - let us consider what happens to Δσ when we change Δε. On the solution
    # surface, we can locally consider Δσ as a function of Δε:
    #
    #   Δσ = Δσ(Δε)
    #
    # Taking this into account, we differentiate both sides of  r = 0  w.r.t. Δε:
    #
    #   dr/d(Δε) = d(0)/d(Δε)
    #
    # which yields, by applying the chain rule on the LHS:
    #
    #   ∂r/∂(Δε) + ∂r/∂(Δσ) d(Δσ)/d(Δε) = 0
    #
    # Solving for d(Δσ)/d(Δε) now yields:
    #
    #   d(Δσ)/d(Δε) = -∂r/∂(Δσ) \ ∂r/∂(Δε)
    #
    # which we can compute as:
    #
    #   d(Δσ)/d(Δε) = -∂r/∂σ_new \ ∂r/∂(Δε)
    #
    # This completes the solution for the jacobian of the simple two-variable
    # case. We can extend the same strategy to compute the jacobian for our
    # actual problem. At a solution point, the residual equation is:
    #
    #   r(Δε, Δσ, ΔR, ΔX1, ΔX2, ΔX3) = 0
    #
    # Packing the state variables into the vector  x ≡ [σ R X1 X2 X3]  (with tensors
    # encoded into Voigt notation), we can rewrite this as:
    #
    #   r(Δε, Δx) = 0
    #
    # Proceeding as before, we differentiate both sides w.r.t. Δε:
    #
    #   dr/d(Δε) = d(0)/d(Δε)
    #
    # Considering Δx as a function of Δε (locally, on the solution surface),
    # and applying the chain rule, we have:
    #
    #   ∂r/∂(Δε) + ∂r/∂(Δx) d(Δx)/d(Δε) = 0
    #
    # Solving for d(Δx)/d(Δε) (which contains d(Δσ)/d(Δε) in its [1:6, 1:6] block) yields:
    #
    #   d(Δx)/d(Δε) = -∂r/∂(Δx) \ ∂r/∂(Δε)
    #
    # which we can compute as:
    #
    #   d(Δx)/d(Δε) = -∂r/∂x_new \ ∂r/∂(Δε)
    #
    # So, we can autodiff the algorithm to obtain both RHS terms, if we
    # parameterize the residual function twice: once by x_new (which is
    # already needed for solving the nonlinear equation system), and
    # once by Δε (keeping all other quantities constant).
    #
    # Note this is slightly expensive. ∂r/∂x_new is a 25×25 matrix,
    # and ∂r/∂(Δε) is 25×6. So essentially, to obtain d(Δx)/d(Δε),
    # from which we can read off d(Δσ)/d(Δε), we must solve six
    # linear equation systems, each of size 25. (In a FEM solver,
    # this must be done for each integration point.)
    #
    # But if we are willing to pay for that, we get the algorithmic jacobian
    # exactly (up to the effects of finite precision arithmetic) - which gives
    # us quadratic convergence in a FEM solver using this material model.

    # Residual function. (Actual implementation in `r!`, below.)
    #
    # This variant is for solving the equation system. F is output, x is filled by NLsolve.
    # The solution is x = x* such that g(x*) = 0.
    #
    # This is a mutating function for performance reasons.
    #
    # Parameterized by the whole new state x_new.
    # We can also autodiff this at the solution point to obtain ∂r/∂x_new.
    function rx!(F::V, x::V) where V <: AbstractVector{<:Real}  # x = new state
        # IMPORTANT: Careful here not to overwrite cell variables
        # from the outer scope. (Those variables hold the *old* values
        # at the start of the timestep.) Either use a new name, or use
        # the `local` annotation.
        stress_new, R_new, X1_new, X2_new, X3_new = state_from_vector(x)
        r!(F, stress_new, R_new, X1_new, X2_new, X3_new, dd.strain, temperature + dtemperature)
        return nothing
    end

    # Residual parameterized by Δε, for algorithmic jacobian computation.
    # Autodiff this (w.r.t. strain) at the solution point to get ∂r/∂(Δε).
    #
    # This we only need to compute once per timestep, so this allocates
    # the output array.
    #
    # The quantity w.r.t. which the function is to be autodiffed must be a
    # parameter, so `ForwardDiff` can promote it to use dual numbers.
    # So `dstrain` must be a parameter. But the old state (from `material`)
    # is used in several internal computations above. So the value of `dstrain`
    # given to this routine **must be** `material.ddrivers.strain`.
    #
    # We also need to pass the new state (the solution point) to the underlying
    # residual function `r!`. Use the partial application pattern to provide that:
    #
    #   r(dstrain) = rdstrain(stress, R, X1, X2, X3, dstrain)
    #   ForwardDiff.jacobian(r, tovoigt(dstrain))
    function rdstrain(stress_new::Symm2{<:Real}, R_new::Real,
                      X1_new::Symm2{<:Real}, X2_new::Symm2{<:Real}, X3_new::Symm2{<:Real},
                      x::V) where V <: AbstractVector{<:Real}  # x = dstrain
        F = similar(x, eltype(x), (25,))
        # We don't bother with offdiagscale, since this Voigt conversion is just a marshaling.
        # All computations are performed in tensor format.
        dstrain = fromvoigt(Symm2, x)
        r!(F, stress_new, R_new, X1_new, X2_new, X3_new, dstrain, temperature + dtemperature)
        return F
    end

    function rtemperature(stress_new::Symm2{<:Real}, R_new::Real,
                          X1_new::Symm2{<:Real}, X2_new::Symm2{<:Real}, X3_new::Symm2{<:Real},
                          x::V) where V <: AbstractVector{<:Real}  # x = temperature_new
        F = similar(x, eltype(x), (25,))
        temperature_new = x[1]
        r!(F, stress_new, R_new, X1_new, X2_new, X3_new, dd.strain, temperature_new)
        return F
    end

    # TODO: decouple integrator

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
    # `F` is output, length 25.
    function r!(F::V, stress_new::Symm2{<:Real}, R_new::Real,
                X1_new::Symm2{<:Real}, X2_new::Symm2{<:Real}, X3_new::Symm2{<:Real},
                dstrain::Symm2{<:Real}, temperature_new::Real) where {V <: AbstractVector{<:Real}}
        # This stuff must be done here so we can autodiff it w.r.t. temperature_new.
        thermal_strainf(theta) = thermal_strain_tensor(alphaf, theta0, theta)
        # thermal_strain_derivative(theta) = gradient(thermal_strainf, theta)
        # thermal_dstrain = thermal_strain_derivative(temperature_new) * (temperature_new - temperature)
        thermal_dstrain = thermal_strainf(temperature_new) - thermal_strainf(temperature)

        Df(theta) = elasticity_tensor(Ef, nuf, theta)  # dσ/dε, i.e. ∂σij/∂εkl
        dDdthetaf(theta) = gradient(Df, theta)
        D = Df(temperature_new)
        dDdtheta = dDdthetaf(temperature_new)

        dotp = dotpf(stress_new, R_new, X1_new, X2_new, X3_new, temperature_new)
        n = nf(stress_new, R_new, X1_new, X2_new, X3_new, temperature_new)

        local dtemperature = temperature_new - temperature

        dp = dotp * dtime
        plastic_dstrain = dp * n
        elastic_dstrain = dstrain - plastic_dstrain - thermal_dstrain
        elastic_strain = elastic_strain_old + elastic_dstrain
        tovoigt!(view(F, 1:6),
                 stress_new - stress
                 - dcontract(D, elastic_dstrain)
                 - dcontract(dDdtheta, elastic_strain) * dtemperature)

        # Reijo's equations (37) and (43), for exponentially saturating
        # isotropic hardening, are:
        #
        #   Kiso = Kiso∞ (1 - exp(-hiso κiso / Kiso∞))
        #   κiso' = 1 / tvp <fhat / σ0>^p
        #
        # Our equation for R in the case without thermal effects, where
        # Q and b are constant, is:
        #
        #   R' = b (Q - R) p'
        #
        # We identify (LHS Reijo's notation; RHS Materials.jl notation):
        #
        #   Kiso = R, κiso = p, σ0 = Kn, p = nn
        #   TODO: is fhat our f? Looks a bit different.
        #
        # So in the notation used in Materials.jl:
        #
        #   R = R∞ (1 - exp(-hiso p / R∞))
        #   p' = 1 / tvp <fhat / Kn>^nn
        #
        # which leads to
        #
        #   R' = -R∞ * (-hiso p'/ R∞) exp(-hiso p / R∞)
        #      = hiso exp(-hiso p / R∞) p'
        #      = hiso (1 - R / R∞) p'
        #      = hiso p' / R∞ (R∞ - R)
        #      = (hiso / R∞) (R∞ - R) p'
        #      ≡ b (Q - R) p'
        #
        # where
        #
        #   Q := R∞
        #   b := hiso / R∞
        #
        # Thus we can write
        #
        #   R = Q (1 - exp(-b p))
        #
        #
        # Now, if we model thermal effects by  Q = Q(θ),  b = b(θ),  we have
        #
        #   R' = ∂Q/∂θ θ' (1 - exp(-b p)) - Q (-b p)' exp(-b p)
        #      = ∂Q/∂θ θ' (1 - exp(-b p)) + Q (∂b/∂θ θ' p + b p') exp(-b p)
        #
        # Observe that
        #
        #   Q exp(-b p) = Q - R
        #   1 - exp(-b p) = R / Q
        #
        # so we can write
        #
        #   R' = (∂Q/∂θ / Q) θ' R + (∂b/∂θ θ' p + b p') (Q - R)
        #      = b (Q - R) p' + ((∂Q/∂θ / Q) R + ∂b/∂θ (Q - R) p) θ'
        #
        # on the condition that Q ≠ 0.
        #
        # But that's a disaster when Q = 0 (no isotropic hardening),
        # so let's use  R / Q = 1 - exp(-b p)  to obtain
        #
        #   R' = b (Q - R) p' + (∂Q/∂θ (1 - exp(-b p)) + ∂b/∂θ (Q - R) p) θ'
        #
        # which is the form we use here.
        #
        Q = Qf(temperature_new)
        b = bf(temperature_new)
        dQdtheta = gradient(Qf, temperature_new)
        dbdtheta = gradient(bf, temperature_new)
        cumeq_new = v.cumeq + dp
        # TODO: p (cumeq) accumulates too much error to be usable here.
        # TODO: As t increases, R will drift until the solution becomes nonsense.
        # TODO: So for now, we approximate ∂Q/∂θ = ∂b/∂θ = 0 to eliminate terms
        # TODO: that depend on p. (p' is fine; computed afresh every timestep.)
        # F[7] = R_new - R - (b*(Q - R_new) * dp
        #                     + (dQdtheta * (1 - exp(-b * cumeq_new))
        #                        + dbdtheta * (Q - R_new) * cumeq_new)
        #                     * dtemperature)
        F[7] = R_new - R - b*(Q - R_new) * dp

        # Reijo's equations (44) and (38):
        #
        #   κk' = εp' - 1 / tvp <fhat / σ0>^p (3 / Kk∞) Kk
        #   Kk = 2/3 hk κk
        #
        # In Materials.jl, we have:
        #
        #   εp' = p' n
        #   p' = 1 / tvp <fhat / Kn>^nn
        #
        # so (in a mixed abuse of notation)
        #
        #   κk' = p' n - p' (3 / Kk∞) Kk
        #       = p' (n - (3 / Kk∞) Kk)
        #
        # In the case without thermal effects, hk is a constant, so:
        #
        #   Kk' = 2/3 hk κk'
        #       = 2/3 hk p' (n - (3 / Kk∞) Kk)
        #       = p' (2/3 hk n - (2 hk / Kk∞) Kk)
        #
        # The equation used in Materials.jl is:
        #
        #   Xk' = p' (2/3 Ck n - Dk Xk)
        #
        # so we identify
        #
        #   Xk = Kk,  Ck = hk,  Dk = 2 hk / Kk∞
        #
        #
        # Now let us model thermal effects by  Ck = Ck(θ),  Dk = Dk(θ).
        # In Materials.jl notation, we have:
        #
        #   Kk∞ = 2 Ck / Dk
        #
        # when Dk ≠ 0, so if also Ck ≠ 0, then
        #
        #   3 / Kk∞ = 3/2 Dk / Ck
        #
        # We have:
        #
        #   κk' = p' (n - 3/2 (Dk / Ck) Kk)
        #   Kk = 2/3 Ck κk
        #
        # Differentiating:
        #
        #   Kk' = 2/3 (Ck' κk + Ck κk')
        #       = 2/3 (∂Ck/∂θ θ' κk + Ck κk')
        #       = 2/3 (∂Ck/∂θ θ' κk + p' (Ck n - 3/2 Dk Kk))
        #       = 2/3 ∂Ck/∂θ θ' κk + p' (2/3 Ck n - Dk Kk)
        #
        # To avoid the need to track the internal variables κk as part of the
        # problem state, we can use:
        #
        #   Kk = 2/3 Ck κk
        #
        # So whenever Ck ≠ 0,
        #
        #   κk = 3/2 Kk / Ck
        #
        # Final result:
        #
        # Xk' = 2/3 ∂Ck/∂θ θ' κk + p' (2/3 Ck n - Dk Xk)
        #     = 2/3 ∂Ck/∂θ θ' (3/2 Xk / Ck) + p' (2/3 Ck n - Dk Xk)
        #     = (∂Ck/∂θ / Ck) Xk θ' + p' (2/3 Ck n - Dk Xk)
        #
        # ------------------------------------------------------------
        #
        # We identified  Ck = hk,  Dk = 2 hk / Kk∞.  The special case
        # Ck(θ) = Dk(θ) ≡ 0  corresponds to  hk = 0,  Kk∞ → +∞. Then we have:
        #
        #   κk' = p' (n - (3 / Kk∞) Kk) → p' n
        #   Kk' ≡ 0
        #
        # Also, because  Kk = 2/3 Ck κk,  we have  Kk ≡ 0.
        #
        # In this case we can discard the internal variables κk, because they
        # only contribute to Kk.
        #
        # ------------------------------------------------------------
        #
        # Incremental form:
        #
        # ΔXk = 2/3 ∂Ck/∂θ Δθ κk + Δp (2/3 Ck n - Dk Xk)
        #     = 2/3 ∂Ck/∂θ Δθ (3/2 Xk / Ck) + Δp (2/3 Ck n - Dk Xk)
        #     = (∂Ck/∂θ / Ck) Xk Δθ + Δp (2/3 Ck n - Dk Xk)
        #
        C1 = C1f(temperature_new)
        dC1dtheta = gradient(C1f, temperature_new)
        logdiff1 = (C1 != 0.0) ? (dC1dtheta / C1) : 0.0
        D1 = D1f(temperature_new)
        C2 = C2f(temperature_new)
        dC2dtheta = gradient(C2f, temperature_new)
        logdiff2 = (C2 != 0.0) ? (dC2dtheta / C2) : 0.0
        D2 = D2f(temperature_new)
        C3 = C3f(temperature_new)
        dC3dtheta = gradient(C3f, temperature_new)
        logdiff3 = (C3 != 0.0) ? (dC3dtheta / C3) : 0.0
        D3 = D3f(temperature_new)
        tovoigt!(view(F,  8:13), X1_new - X1 - (logdiff1 * X1_new * dtemperature
                                                + dp*(2.0/3.0*C1*n - D1*X1_new)))
        tovoigt!(view(F, 14:19), X2_new - X2 - (logdiff2 * X2_new * dtemperature
                                                + dp*(2.0/3.0*C2*n - D2*X2_new)))
        tovoigt!(view(F, 20:25), X3_new - X3 - (logdiff3 * X3_new * dtemperature
                                                + dp*(2.0/3.0*C3*n - D3*X3_new)))
        return nothing
    end

    return rx!, rdstrain, rtemperature
end

end
