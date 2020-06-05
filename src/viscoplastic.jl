# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

# Viscoplastic material. See:
#   http://www.solid.iei.liu.se/Education/TMHL55/TMHL55_lp1_2010/lecture_notes/plasticity_flow_rule_isotropic_hardening.pdf
#   http://mms.ensmp.fr/msi_paris/transparents/Georges_Cailletaud/2013-GC-plas3D.pdf

POTENTIAL_FUNCTIONS = [:norton, :bingham]

@with_kw mutable struct ViscoPlasticDriverState <: AbstractMaterialState
    time::Float64 = zero(Float64)
    strain::Symm2 = zero(Symm2{Float64})
end

@with_kw struct ViscoPlasticParameterState <: AbstractMaterialState
    youngs_modulus::Float64 = zero(Float64)
    poissons_ratio::Float64 = zero(Float64)
    yield_stress::Float64 = zero(Float64)
    # Settings for the plastic potential.
    potential::Symbol = :norton
    params::Vector{Float64} = [180.0e3, 0.92]  # If potential=:norton, this is [K, n].
    @assert potential in POTENTIAL_FUNCTIONS  # https://mauro3.github.io/Parameters.jl/dev/manual/
end

@with_kw struct ViscoPlasticVariableState <: AbstractMaterialState
    stress::Symm2 = zero(Symm2{Float64})
    plastic_strain::Symm2 = zero(Symm2{Float64})
    cumeq::Float64 = zero(Float64)
    jacobian::Symm4 = zero(Symm4{Float64})
end

@with_kw mutable struct ViscoPlastic <: AbstractMaterial
    drivers::ViscoPlasticDriverState = ViscoPlasticDriverState()
    ddrivers::ViscoPlasticDriverState = ViscoPlasticDriverState()
    variables::ViscoPlasticVariableState = ViscoPlasticVariableState()
    variables_new::ViscoPlasticVariableState = ViscoPlasticVariableState()
    parameters::ViscoPlasticParameterState = ViscoPlasticParameterState()
    dparameters::ViscoPlasticParameterState = ViscoPlasticParameterState()
end

"""Norton rule.

`params = [K, n]`.

`f` is the yield function.

`stress` is passed through to `f` as its only argument,
so its storage format must be whatever `f` expects.
"""
function norton_plastic_potential(stress::Symm2{<:Number}, params::AbstractVector{<:Number}, f::Function)
    K, n = params
    return K/(n+1) * (f(stress) / K)^(n + 1)
end

"""Bingham rule.

`params = [eta]`.

`f` and `stress` like in `norton_plastic_potential`.
"""
function bingham_plastic_potential(stress::Symm2{<:Number}, params::AbstractVector{<:Number}, f::Function)
    eta = params[1]
    return 0.5 * (f(stress) / eta)^2
end

"""
    state_to_vector(sigma::Symm2{<:Real})

Adaptor for `nlsolve`. Marshal the problem state into a `Vector`.
"""
@inline function state_to_vector(sigma::Symm2{<:Real})
    return [tovoigt(sigma); 0.0]  # The extra zero is padding to make the input/output shapes of g! match.
end

"""
    state_from_vector(x::AbstractVector{S}) where S <: Real

Adaptor for `nlsolve`. Unmarshal the problem state from a `Vector`.
"""
@inline function state_from_vector(x::AbstractVector{S}) where S <: Real
    sigma = fromvoigt(Symm2{S}, @view x[1:6])
    # The padding (component 7) is unused, so we just ignore it here.
    return sigma
end

"""
    integrate_material!(material::ViscoPlastic)

Viscoplastic material, with switchable plastic potential. No hardening.
"""
function integrate_material!(material::ViscoPlastic)
    p = material.parameters
    v = material.variables
    dd = material.ddrivers
    d = material.drivers

    E = p.youngs_modulus
    nu = p.poissons_ratio
    R0 = p.yield_stress
    lambda, mu = lame(E, nu)

    @unpack strain, time = d
    dstrain = material.ddrivers.strain
    dtime = material.ddrivers.time
    @unpack stress, plastic_strain, cumeq = v

    # elastic part
    jacobian = isotropic_elasticity_tensor(lambda, mu)  # dσ/dε, i.e. ∂σij/∂εkl
    stress += dcontract(jacobian, dstrain)  # add the elastic stress increment, get the elastic trial stress

    f(sigma) = sqrt(1.5)*norm(dev(sigma)) - R0  # von Mises yield function

    if f(stress) > 0
        params = mat.parameters.params  # parameters for the plastic potential function

        if mat.parameters.potential == :norton
            psi(sigma) -> norton_plastic_potential(sigma, params, f)
        elseif mat.parameters.potential == :bingham
            psi(sigma) -> bingham_plastic_potential(sigma, params, f)
        else
            @assert false  # cannot happen
        end

        dpsi_dsigma(sigma) = ForwardDiff.jacobian(psi, sigma)

        # Compute the residual. F is output, x is filled by NLsolve.
        # The solution is x = x* such that g(x*) = 0.
        function g!(F, x)
            dsigma = state_from_vector(x)  # tentative new value from nlsolve
            # Evolution equation in delta form:
            #
            #   Δσ = D : (Δε - (∂ψ/∂σ)(σ_new) Δt)
            #
            # where
            #
            #   Δσ ≡ σ_new - σ_old
            #
            # Then move terms to get the standard form, (stuff) = 0.
            #
            sigma_new = stress + dsigma
            F[1:6] = dsigma - dcontract(jacobian, (dstrain - (dpsi_dsigma(sigma_new) * dtime)))
            # Constraint: the solution is on the yield surface:
            #   f(σ_new) = 0
            F[end] = f(sigma_new)
        end

        x0 = state_to_vector(dstress)
        res = nlsolve(g!, x0, autodiff=:forward)
        converged(res) || error("Nonlinear system of equations did not converge!")
        dsigma = state_from_vector(res.zero)

        stress += dsigma

        plastic_dstrain = dpsi_dsigma(stress)
        dp = norm(plastic_dstrain)  # TODO: verify. Do we need to divide by sqrt(1.5) to match the other models?

        plastic_strain += plastic_dstrain
        cumeq += dp  # cumulative equivalent plastic strain (note dp ≥ 0)

        # TODO: Verify. This comes from chaboche.jl, but the algorithm looks general enough. Comments updated.
        #
        # TODO: I still don't see where the minus sign comes from. Using the chain rule
        # TODO: and the inverse function theorem, there should be no minus sign.
        #
        # Compute the new Jacobian, accounting for the plastic contribution. Because
        #   x ≡ [σ 0]
        # we have
        #   (dx/dε)[1:6,1:6] = dσ/dε
        # for which we can compute the LHS as follows:
        #   dx/dε = dx/dr dr/dε = inv(dr/dx) dr/dε ≡ (dr/dx) \ (dr/dε)
        # where r = r(x) is the residual, given by the function g!. AD can get us dr/dx automatically,
        # the other factor we will have to supply manually.
        drdx = ForwardDiff.jacobian(debang(g!), x)  # Array{7, 7}
        drde = zeros((length(x),6))                 # Array{7, 6}
        drde[1:6, 1:6] = -tovoigt(jacobian)  # (negative of the) elastic Jacobian. Follows from the defn. of g!.
        jacobian = fromvoigt(Symm4, (drdx\drde)[1:6, 1:6])
    end

    variables_new = ViscoPlasticVariableState(stress = stress,
                                              plastic_strain = plastic_strain,
                                              cumeq = cumeq,
                                              jacobian = jacobian)
    material.variables_new = variables_new
    return nothing
end
