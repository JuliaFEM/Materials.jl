# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE
using LinearAlgebra
# using ForwardDiff
using NLsolve

# http://www.solid.iei.liu.se/Education/TMHL55/TMHL55_lp1_2010/lecture_notes/plasticity_flow_rule_isotropic_hardening.pdf
# http://mms.ensmp.fr/msi_paris/transparents/Georges_Cailletaud/2013-GC-plas3D.pdf

#####################################
# Viscoplastic material definitions #
#####################################
mutable struct ViscoPlastic <: AbstractMaterial
    # Material parameters
    youngs_modulus :: Float64
    poissons_ratio :: Float64
    yield_stress :: Float64
    potential :: Symbol
    params :: Array{Float64, 1}

    # Internal state variables
    plastic_strain :: Vector{Float64}
    dplastic_strain :: Vector{Float64}
    plastic_multiplier :: Float64
    dplastic_multiplier :: Float64

end

POTENTIAL_FUNCTIONS = [:norton]

function ViscoPlastic(potential::Symbol, params :: Array{Float64,1})
    if !any(POTENTIAL_FUNCTIONS .== potential)
        error("Potential $potential not found!")
    end
    youngs_modulus = 0.0
    poissons_ratio = 0.0
    yield_stress = 0.0
    plastic_strain = zeros(6)
    dplastic_strain = zeros(6)
    plastic_multiplier = 0.0
    dplastic_multiplier = 0.0
    dt = 0.0
    return ViscoPlastic(youngs_modulus, poissons_ratio, yield_stress, potential, params,
                        plastic_strain, dplastic_strain, plastic_multiplier,
                        dplastic_multiplier)
end

function ViscoPlastic()
    youngs_modulus = 0.0
    poissons_ratio = 0.0
    yield_stress = 0.0
    plastic_strain = zeros(6)
    dplastic_strain = zeros(6)
    plastic_multiplier = 0.0
    dplastic_multiplier = 0.0
    dt = 0.0
    potential = :norton
    params = [180.0e3, 0.92]
    return ViscoPlastic(youngs_modulus, poissons_ratio, yield_stress, potential, params,
                        plastic_strain, dplastic_strain, plastic_multiplier,
                        dplastic_multiplier)
end

""" Double contraction
"""
function double_contraction(x,y)
    return sum(x.*y)
end

""" Deviatoric stress tensor
"""
function deviatoric_stress(stress)
    stress_tensor = [   stress[1] stress[4] stress[6];
                        stress[4] stress[2] stress[5];
                        stress[6] stress[5] stress[3]]
    stress_dev = stress_tensor - 1/3 * tr(stress_tensor) * Matrix(1.0I, 3, 3)
    return stress_dev
end

function J_2_stress(stress)
    s = deviatoric_stress(stress)
    return 1/2 * double_contraction(s,s)
end

""" Equivalent stress
"""
function equivalent_stress(stress)
    J_2 = J_2_stress(stress)
    return sqrt(3 * J_2)
end

""" Von mises yield function
"""
function von_mises_yield(stress, stress_y)
    return equivalent_stress(stress) - stress_y
end

""" Norton rule
"""
function norton_plastic_potential(stress, params, f)
    K = params[1]
    n = params[2]
    J = f(stress)
    return K/(n+1) * (J / K) ^ (n + 1)
end

function bingham_plastic_potential(stress, params, f)
    eta = params[1]
    return 0.5 * (f(stress) / eta) ^2
end

""" Analytically derivated norton rule
"""
function dNortondStress(stress, params, f)
    K = params[1]
    n = params[2]
    f_ = f(stress)
    stress_v = equivalent_stress(stress)
    stress_dev_vec = deviatoric_stress(stress)
    sol_mat = (f_ / K) ^ n * 3 / 2 * stress_dev_vec / stress_v
    return [sol_mat[1,1], sol_mat[2,2], sol_mat[3,3],
            2*sol_mat[1,2], 2*sol_mat[2,3], 2*sol_mat[1,3]]
end

function find_root(f, df, x; max_iter=50, norm_acc=1e-9)
    converged = false
    for i=1:max_iter
        dx = -df(x) \ f(x)
        x += dx
        norm(dx) < norm_acc && (converged = true; break)
    end
    converged || error("No convergence in radial return!")
    return x
end

function integrate_material!(material::Material{ViscoPlastic})
    mat = material.properties

    E = mat.youngs_modulus
    nu = mat.poissons_ratio
    potential = mat.potential
    params = mat.params
    mu = E/(2.0*(1.0+nu))
    lambda = E*nu/((1.0+nu)*(1.0-2.0*nu))

    stress = material.stress
    strain = material.strain
    dstress = material.dstress
    dstrain = material.dstrain
    dt = material.dtime
    D = material.jacobian

    dedt = dstrain ./ dt

    fill!(D, 0.0)
    D[1,1] = D[2,2] = D[3,3] = 2.0*mu + lambda
    D[4,4] = D[5,5] = D[6,6] = mu
    D[1,2] = D[2,1] = D[2,3] = D[3,2] = D[1,3] = D[3,1] = lambda

    dstress[:] .= D * dedt .* dt
    stress_tr = stress + dstress
    f = x -> von_mises_yield(x, mat.yield_stress)
    yield = f(stress_tr)

    if yield <= 0
        fill!(mat.dplastic_strain, 0.0)
        return nothing
    else
        if potential == :norton
            x0 = vec([dstress; 0.0])

            function g!(F, x)
                dsigma = x[1:6]
                F[1:6] = dsigma - D * (dedt - dNortondStress(stress + dsigma, params, f)) .* dt
                F[end] = von_mises_yield(stress + dsigma, mat.yield_stress)
            end

            res = nlsolve(g!, x0)
            dsigma = res.zero[1:6]
            dstrain_pl = dNortondStress(stress + dsigma, params, f)

        end
        mat.dplastic_strain = dstrain_pl
        dstress[:] .= D*(dedt - dstrain_pl) .* dt
        return nothing
    end

    return nothing

end
