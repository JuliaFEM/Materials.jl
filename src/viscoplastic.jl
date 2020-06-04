# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

# Viscoplastic material. See:
#   http://www.solid.iei.liu.se/Education/TMHL55/TMHL55_lp1_2010/lecture_notes/plasticity_flow_rule_isotropic_hardening.pdf
#   http://mms.ensmp.fr/msi_paris/transparents/Georges_Cailletaud/2013-GC-plas3D.pdf

using LinearAlgebra  # in Julia 1.0+, the I matrix lives here.
using NLsolve

POTENTIAL_FUNCTIONS = [:norton]

# TODO: To conform with the API of the other models, viscoplastic still needs:
#   - drivers, ddrivers, variables, variables_new, parameters, dparameters
#   - ViscoPlasticDriverState: maybe time, strain?
#   - ViscoPlasticParameterState: youngs_modulus, poissons_ratio, yield_stress, potential, params
#   - ViscoPlasticVariableState: stress, plastic_strain, cumeq, jacobian
@with_kw mutable struct ViscoPlastic <: AbstractMaterial
    # Material parameters
    youngs_modulus::Float64 = zero(Float64)
    poissons_ratio::Float64 = zero(Float64)
    yield_stress::Float64 = zero(Float64)
    potential::Symbol = :norton
    # Parameters for potential. If potential=:norton, this is [K, n].
    params::Vector{Float64} = [180.0e3, 0.92]
    # Internal state variables
    plastic_strain::Vector{Float64} = zeros(6)  # TODO: never used?
    dplastic_strain::Vector{Float64} = zeros(6)
    plastic_multiplier::Float64 = zero(Float64)  # TODO: never used, remove?
    dplastic_multiplier::Float64 = zero(Float64)  # TODO: never used, remove?
    # https://mauro3.github.io/Parameters.jl/dev/manual/
    @assert potential in POTENTIAL_FUNCTIONS
end

# TODO: Rewrite this module to use regular tensor notation.
# Could then just use the functions from the Tensors package.
"""Double contraction of rank-2 tensors stored in an array format."""
function dcontract(x::A, y::A) where A <: AbstractArray{<:Number, 2}
    return sum(x .* y)
end

"""Deviatoric part of rank-2 tensor.

Input in Voigt format. Output as a 3×3 array`.

Order of components is:

    t = [v[1] v[4] v[6];
         v[4] v[2] v[5];
         v[6] v[5] v[3]]
"""
function dev(v::AbstractVector{<:Number})
    t = [v[1] v[4] v[6];
         v[4] v[2] v[5];
         v[6] v[5] v[3]]
    # TODO: Unify storage order with the Julia standard:
    #   fromvoigt(SymmetricTensor{2,3}, Array(1:6))
    #
    #   1 6 5
    #   6 2 4
    #   5 4 3
    #
    # Changing this affects how to read sol_mat in dNortondStress.
    return t - 1/3 * tr(t) * I
end

"""Equivalent stress."""
function equivalent_stress(stress::AbstractVector{<:Number})
    s = dev(stress)
    J_2 = 1/2 * dcontract(s,s)
    return sqrt(3 * J_2)
end

"""Norton rule.

`params = [K, n]`.

`f` is the yield function.

`stress` is passed through to `f` as its only argument,
so its storage format must be whatever `f` expects.
"""
function norton_plastic_potential(stress, params::AbstractVector{<:Number}, f::Function)
    K = params[1]
    n = params[2]
    return K/(n+1) * (f(stress) / K)^(n + 1)
end

"""Bingham rule.

`params = [eta]`.

`f` and `stress` like in `norton_plastic_potential`.
"""
function bingham_plastic_potential(stress, params::AbstractVector{<:Number}, f::Function)
    eta = params[1]
    return 0.5 * (f(stress) / eta)^2
end

"""Jacobian of the Norton rule."""
function dNortondStress(stress::AbstractVector{<:Number}, params::AbstractVector{<:Number}, f::Function)
    # using SymPy
    # @vars K n s
    # @symfuns f
    # norton = K / (n + 1) * (f(s) / K)^(n + 1)
    # simplify(diff(norton, s))
    #   --> (f(s) / K)^n df/ds
    #
    # ...so this only works if `f` is the von Mises yield function.
    #
    # K = params[1]
    # n = params[2]
    # stress_v = equivalent_stress(stress)
    # stress_dev = dev(stress)
    # sol_mat = (f(stress) / K)^n * 3.0/2.0 * stress_dev / stress_v
    # return [sol_mat[1,1], sol_mat[2,2], sol_mat[3,3],
    #         2*sol_mat[1,2], 2*sol_mat[2,3], 2*sol_mat[1,3]]

    # Instead, let's use AD:
    pot(stress) = norton_plastic_potential(stress, params, f)
    return ForwardDiff.jacobian(pot, stress)
end

function integrate_material!(material::ViscoPlastic)
    mat = material.properties

    E = mat.youngs_modulus
    nu = mat.poissons_ratio
    R0 = mat.yield_stress
    potential = mat.potential
    params = mat.params
    lambda, mu = lame(E, nu)

    # TODO: these fields don't exist!
    stress = material.stress
    strain = material.strain
    dstress = material.dstress
    dstrain = material.dstrain
    dt = material.dtime
    # D = material.jacobian  # TODO: this field doesn't exist!

    dedt = dstrain ./ dt

    # This looks like a rank-4 tensor in a Voigt matrix format...?
    # fill!(D, 0.0)
    # D[1,1] = D[2,2] = D[3,3] = 2.0*mu + lambda
    # D[4,4] = D[5,5] = D[6,6] = mu
    # D[1,2] = D[2,1] = D[2,3] = D[3,2] = D[1,3] = D[3,1] = lambda
    #
    # Let's find out. In the REPL:
    #   S = SymmetricTensor{2, 3}(1:6)
    #   tovoigt(S)
    # So the index mapping is:
    #   Voigt  rank-2
    #   1      11
    #   2      22
    #   3      33
    #   4      23
    #   5      13
    #   6      12
    # So this is just the isotropic elasticity tensor C:
    #   Cijkl = λ δij δkl + μ (δik δjl + δil δjk)
    # with the strain stored in a Voigt format:
    #   [εxx, εyy, εzz, εyz, εxz, εxy]  (ordering of shear components doesn't matter here)
    #
    # So let's just:
    D = tovoigt(isotropic_elasticity_tensor(lambda, mu))

    """von Mises yield function."""
    function f(stress::AbstractVector{<:Number})  # stress in Voigt format
        return equivalent_stress(stress) - R0
    end

    dstress[:] .= D * dedt .* dt
    stress_elastic_trial = stress + dstress

    if f(stress_elastic_trial) <= 0  # not in plastic region
        fill!(mat.dplastic_strain, 0.0)
        return nothing
    end

    if potential == :norton
        pot(stress) -> norton_plastic_potential(stress, params, f)
    else
        @assert false  # cannot happen (unless someone mutates the struct contents)
    end

    dpotdstress(stress) = ForwardDiff.jacobian(pot, stress)

    # The nonlinear equation system, (stuff) = 0
    function g!(F, x)
        dsigma = x[1:6]
        F[1:6] = dsigma - D * (dedt - dpotdstress(stress + dsigma)) .* dt
        F[end] = f(stress + dsigma)
    end

    x0 = vec([dstress; 0.0])
    res = nlsolve(g!, x0, autodiff=:forward)
    converged(res) || error("Nonlinear system of equations did not converge!")
    dsigma = res.zero[1:6]

    dstrain_plastic = dpotdstress(stress + dsigma)
    mat.dplastic_strain = dstrain_plastic
    dstress[:] .= D * (dedt - dstrain_plastic) .* dt
    return nothing
end
