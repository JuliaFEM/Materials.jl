# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

module PerfectPlasticModule

using LinearAlgebra, ForwardDiff, Tensors, NLsolve, Parameters

import ..AbstractMaterial, ..AbstractMaterialState
import ..Utilities: Symm2, Symm4, isotropic_elasticity_tensor, IS, ID, lame
import ..integrate_material!  # for method extension

# parametrically polymorphic for any type representing ℝ
export GenericPerfectPlastic, GenericPerfectPlasticDriverState, GenericPerfectPlasticParameterState, GenericPerfectPlasticVariableState

# specialization for Float64
export PerfectPlastic, PerfectPlasticDriverState, PerfectPlasticParameterState, PerfectPlasticVariableState

@with_kw mutable struct GenericPerfectPlasticDriverState{T <: Real} <: AbstractMaterialState
    time::T = zero(T)
    strain::Symm2{T} = zero(Symm2{T})
end

@with_kw struct GenericPerfectPlasticParameterState{T <: Real} <: AbstractMaterialState
    youngs_modulus::T = zero(T)
    poissons_ratio::T = zero(T)
    yield_stress::T = zero(T)
end

@with_kw struct GenericPerfectPlasticVariableState{T <: Real} <: AbstractMaterialState
    stress::Symm2{T} = zero(Symm2{T})
    plastic_strain::Symm2{T} = zero(Symm2{T})
    cumeq::T = zero(T)
    jacobian::Symm4{T} = zero(Symm4{T})
end

@with_kw mutable struct GenericPerfectPlastic{T <: Real} <: AbstractMaterial
    drivers::GenericPerfectPlasticDriverState{T} = GenericPerfectPlasticDriverState{T}()
    ddrivers::GenericPerfectPlasticDriverState{T} = GenericPerfectPlasticDriverState{T}()
    variables::GenericPerfectPlasticVariableState{T} = GenericPerfectPlasticVariableState{T}()
    variables_new::GenericPerfectPlasticVariableState{T} = GenericPerfectPlasticVariableState{T}()
    parameters::GenericPerfectPlasticParameterState{T} = GenericPerfectPlasticParameterState{T}()
    dparameters::GenericPerfectPlasticParameterState{T} = GenericPerfectPlasticParameterState{T}()
end

PerfectPlastic = GenericPerfectPlastic{Float64}
PerfectPlasticDriverState = GenericPerfectPlasticDriverState{Float64}
PerfectPlasticParameterState = GenericPerfectPlasticParameterState{Float64}
PerfectPlasticVariableState = GenericPerfectPlasticVariableState{Float64}

"""
    integrate_material!(material::GenericPerfectPlastic)

Perfect plastic material: no hardening. The elastic region remains centered on the
origin, and retains its original size.
"""
function integrate_material!(material::GenericPerfectPlastic{T}) where T <: Real
    p = material.parameters
    v = material.variables
    dd = material.ddrivers
    d = material.drivers

    E = p.youngs_modulus
    nu = p.poissons_ratio
    lambda, mu = lame(E, nu)
    R0 = p.yield_stress

    # @unpack strain, time = d  # not needed for this material
    dstrain = dd.strain
    dtime = dd.time
    @unpack stress, plastic_strain, cumeq, jacobian = v

    jacobian = isotropic_elasticity_tensor(lambda, mu)  # dσ/dε, i.e. ∂σij/∂εkl
    stress += dcontract(jacobian, dstrain)  # add the elastic stress increment, get the elastic trial stress
    seff_dev = dev(stress)
    f = sqrt(1.5)*norm(seff_dev) - R0  # von Mises yield function; f := J(seff_dev) - Y

    if f > 0.0
        dp = 1.0/(3.0*mu) * f
        n = sqrt(1.5)*seff_dev/norm(seff_dev)  # a (tensorial) unit direction, s.t. 2/3 * (n : n) = 1

        plastic_strain += dp*n
        cumeq += dp  # cumulative equivalent plastic strain (note dp ≥ 0)

        # Perfect plastic material: the stress state cannot be outside the yield surface.
        # Project it back to the yield surface.
        stress -= dcontract(jacobian, dp*n)

        # Compute ∂σij/∂εkl, accounting for the plastic contribution.
        # EE = IS + dp/R0 * (∂σ/∂ε)_e : ((3/2) ID - n ⊗ n)
        EE = IS(T) + dp/R0 * dcontract(jacobian, 1.5*ID(T) - otimes(n,n))  # using the elastic jacobian
        ED = dcontract(inv(EE), jacobian)
        # J = ED - (ED : n) ⊗ (n : ED) / (n : ED : n)
        jacobian = ED - otimes(dcontract(ED, n), dcontract(n, ED)) / dcontract(dcontract(n, ED), n)
    end
    variables_new = GenericPerfectPlasticVariableState(stress=stress,
                                              plastic_strain=plastic_strain,
                                              cumeq=cumeq,
                                              jacobian=jacobian)
    material.variables_new = variables_new
    return nothing
end

end
