# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

module IdealPlasticModule

using LinearAlgebra, ForwardDiff, Tensors, NLsolve, Parameters

import ..AbstractMaterial, ..AbstractMaterialState
import ..Utilities: Symm2, Symm4, isotropic_elasticity_tensor, IS, ID, lame
import ..integrate_material!  # for method extension

export IdealPlastic, IdealPlasticDriverState, IdealPlasticParameterState, IdealPlaticVariableState

@with_kw mutable struct IdealPlasticDriverState <: AbstractMaterialState
    time::Float64 = zero(Float64)
    strain::Symm2 = zero(Symm2{Float64})
end

@with_kw struct IdealPlasticParameterState <: AbstractMaterialState
    youngs_modulus::Float64 = zero(Float64)
    poissons_ratio::Float64 = zero(Float64)
    yield_stress::Float64 = zero(Float64)
end

@with_kw struct IdealPlasticVariableState <: AbstractMaterialState
    stress::Symm2 = zero(Symm2{Float64})
    plastic_strain::Symm2 = zero(Symm2{Float64})
    cumeq::Float64 = zero(Float64)
    jacobian::Symm4 = zero(Symm4{Float64})
end

@with_kw mutable struct IdealPlastic <: AbstractMaterial
    drivers::IdealPlasticDriverState = IdealPlasticDriverState()
    ddrivers::IdealPlasticDriverState = IdealPlasticDriverState()
    variables::IdealPlasticVariableState = IdealPlasticVariableState()
    variables_new::IdealPlasticVariableState = IdealPlasticVariableState()
    parameters::IdealPlasticParameterState = IdealPlasticParameterState()
    dparameters::IdealPlasticParameterState = IdealPlasticParameterState()
end

"""
    integrate_material!(material::IdealPlastic)

Ideal plastic material: no hardening. The elastic region remains centered on the
origin, and retains its original size.
"""
function integrate_material!(material::IdealPlastic)
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
        EE = IS(Float64) + dp/R0 * dcontract(jacobian, 1.5*ID(Float64) - otimes(n,n))  # using the elastic jacobian
        ED = dcontract(inv(EE), jacobian)
        # J = ED - (ED : n) ⊗ (n : ED) / (n : ED : n)
        jacobian = ED - otimes(dcontract(ED, n), dcontract(n, ED)) / dcontract(dcontract(n, ED), n)
    end
    variables_new = IdealPlasticVariableState(stress=stress,
                                              plastic_strain=plastic_strain,
                                              cumeq=cumeq,
                                              jacobian=jacobian)
    material.variables_new = variables_new
end

end
