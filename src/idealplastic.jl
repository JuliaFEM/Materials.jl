# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

module IdealPlasticModule

using LinearAlgebra, ForwardDiff, Tensors, NLsolve, Parameters

import ..AbstractMaterial, ..AbstractMaterialState
import ..Utilities: Symm2, Symm4, isotropic_elasticity_tensor, IS, ID, lame
import ..integrate_material!  # for method extension

# parametrically polymorphic for any type representing ℝ
export GenericIdealPlastic, GenericIdealPlasticDriverState, GenericIdealPlasticParameterState, GenericIdealPlasticVariableState

# specialization for Float64
export IdealPlastic, IdealPlasticDriverState, IdealPlasticParameterState, IdealPlasticVariableState

@with_kw mutable struct GenericIdealPlasticDriverState{T <: Real} <: AbstractMaterialState
    time::T = zero(T)
    strain::Symm2{T} = zero(Symm2{T})
end

@with_kw struct GenericIdealPlasticParameterState{T <: Real} <: AbstractMaterialState
    youngs_modulus::T = zero(T)
    poissons_ratio::T = zero(T)
    yield_stress::T = zero(T)
end

@with_kw struct GenericIdealPlasticVariableState{T <: Real} <: AbstractMaterialState
    stress::Symm2{T} = zero(Symm2{T})
    plastic_strain::Symm2{T} = zero(Symm2{T})
    cumeq::T = zero(T)
    jacobian::Symm4{T} = zero(Symm4{T})
end

@with_kw mutable struct GenericIdealPlastic{T <: Real} <: AbstractMaterial
    drivers::GenericIdealPlasticDriverState{T} = GenericIdealPlasticDriverState{T}()
    ddrivers::GenericIdealPlasticDriverState{T} = GenericIdealPlasticDriverState{T}()
    variables::GenericIdealPlasticVariableState{T} = GenericIdealPlasticVariableState{T}()
    variables_new::GenericIdealPlasticVariableState{T} = GenericIdealPlasticVariableState{T}()
    parameters::GenericIdealPlasticParameterState{T} = GenericIdealPlasticParameterState{T}()
    dparameters::GenericIdealPlasticParameterState{T} = GenericIdealPlasticParameterState{T}()
end

IdealPlastic = GenericIdealPlastic{Float64}
IdealPlasticDriverState = GenericIdealPlasticDriverState{Float64}
IdealPlasticParameterState = GenericIdealPlasticParameterState{Float64}
IdealPlasticVariableState = GenericIdealPlasticVariableState{Float64}

"""
    integrate_material!(material::GenericIdealPlastic)

Ideal plastic material: no hardening. The elastic region remains centered on the
origin, and retains its original size.
"""
function integrate_material!(material::GenericIdealPlastic{T}) where T <: Real
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

        # Ideal plastic material: the stress state cannot be outside the yield surface.
        # Project it back to the yield surface.
        stress -= dcontract(jacobian, dp*n)

        # Compute ∂σij/∂εkl, accounting for the plastic contribution.
        # EE = IS + dp/R0 * (∂σ/∂ε)_e : ((3/2) ID - n ⊗ n)
        EE = IS(T) + dp/R0 * dcontract(jacobian, 1.5*ID(T) - otimes(n,n))  # using the elastic jacobian
        ED = dcontract(inv(EE), jacobian)
        # J = ED - (ED : n) ⊗ (n : ED) / (n : ED : n)
        jacobian = ED - otimes(dcontract(ED, n), dcontract(n, ED)) / dcontract(dcontract(n, ED), n)
    end
    variables_new = GenericIdealPlasticVariableState(stress=stress,
                                              plastic_strain=plastic_strain,
                                              cumeq=cumeq,
                                              jacobian=jacobian)
    material.variables_new = variables_new
    return nothing
end

end
