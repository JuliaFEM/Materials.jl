# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

function isotropic_elasticity_tensor(lambda, mu)
    delta(i,j) = i==j ? 1.0 : 0.0
    g(i,j,k,l) = lambda*delta(i,j)*delta(k,l) + mu*(delta(i,k)*delta(j,l)+delta(i,l)*delta(j,k))
    jacobian = SymmetricTensor{4, 3, Float64}(g)
    return jacobian
end

@with_kw mutable struct IdealPlasticDriverState <: AbstractMaterialState
    time :: Float64 = zero(Float64)
    strain :: SymmetricTensor{2,3} = zero(SymmetricTensor{2,3,Float64})
end

@with_kw struct IdealPlasticParameterState <: AbstractMaterialState
    youngs_modulus :: Float64 = zero(Float64)
    poissons_ratio :: Float64 = zero(Float64)
    yield_stress :: Float64 = zero(Float64)
end

@with_kw struct IdealPlasticVariableState <: AbstractMaterialState
    stress :: SymmetricTensor{2,3} = zero(SymmetricTensor{2,3,Float64})
    plastic_strain :: SymmetricTensor{2,3} = zero(SymmetricTensor{2,3,Float64})
    cumeq :: Float64 = zero(Float64)
    jacobian :: SymmetricTensor{4,3} = zero(SymmetricTensor{4,3,Float64})
end

@with_kw mutable struct IdealPlastic <: AbstractMaterial
    drivers :: IdealPlasticDriverState = IdealPlasticDriverState()
    ddrivers :: IdealPlasticDriverState = IdealPlasticDriverState()
    variables :: IdealPlasticVariableState = IdealPlasticVariableState()
    variables_new :: IdealPlasticVariableState = IdealPlasticVariableState()
    parameters :: IdealPlasticParameterState = IdealPlasticParameterState()
    dparameters :: IdealPlasticParameterState = IdealPlasticParameterState()
end

function integrate_material!(material::IdealPlastic)
    p = material.parameters
    v = material.variables
    dd = material.ddrivers
    d = material.drivers

    E = p.youngs_modulus
    nu = p.poissons_ratio
    mu = E/(2.0*(1.0+nu))
    lambda = E*nu/((1.0+nu)*(1.0-2.0*nu))
    R0 = p.yield_stress

    # @unpack strain, time = d
    dstrain = dd.strain
    dtime = dd.time
    @unpack stress, plastic_strain, cumeq, jacobian = v

    jacobian = isotropic_elasticity_tensor(lambda, mu)
    stress += dcontract(jacobian, dstrain)
    seff_dev = dev(stress)
    stress_v = sqrt(1.5)*norm(seff_dev)
    f = stress_v - R0

    if f>0.0
        n = 1.5*seff_dev/stress_v
        dp = 1.0/(3.0*mu)*(stress_v - R0)
        plastic_strain += dp*n
        cumeq += dp
        stress -= dcontract(jacobian, dp*n)
        jacobian -= otimes(dcontract(jacobian, n), dcontract(n, jacobian))/dcontract(dcontract(n, jacobian), n)
    end
    variables_new = IdealPlasticVariableState(stress=stress,
                                              plastic_strain=plastic_strain,
                                              cumeq=cumeq,
                                              jacobian=jacobian)
    material.variables_new = variables_new
end
