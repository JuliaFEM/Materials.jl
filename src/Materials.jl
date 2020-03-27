# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

module Materials

using LinearAlgebra, ForwardDiff, Tensors, NLsolve, Parameters

abstract type AbstractMaterial end
abstract type AbstractMaterialState end

@generated function Base.:+(state::T, dstate::T) where {T <: AbstractMaterialState}
   expr = [:(state.$p + dstate.$p) for p in fieldnames(T)]
   return :(T($(expr...)))
end

export AbstractMaterial, AbstractMaterialState

function integrate_material!(material::M) where {M<:AbstractMaterial}
    error("One needs to define how to integrate material $M!")
end

function update_material!(material::M) where {M <: AbstractMaterial}
    material.drivers += material.ddrivers
    material.parameters += material.dparameters
    material.variables = material.variables_new
    reset_material!(material)
    return nothing
end

function reset_material!(material::M) where {M <: AbstractMaterial}
    material.ddrivers = typeof(material.ddrivers)()
    material.dparameters = typeof(material.dparameters)()
    material.variables_new = typeof(material.variables_new)()
    return nothing
end

export integrate_material!, update_material!, reset_material!

function isotropic_elasticity_tensor(lambda, mu)
    delta(i,j) = i==j ? 1.0 : 0.0
    g(i,j,k,l) = lambda*delta(i,j)*delta(k,l) + mu*(delta(i,k)*delta(j,l)+delta(i,l)*delta(j,k))
    jacobian = SymmetricTensor{4, 3, Float64}(g)
    return jacobian
end

include("idealplastic.jl")
export IdealPlastic, IdealPlasticDriverState, IdealPlasticParameterState, IdealPlasticVariableState

include("chaboche.jl")
export Chaboche, ChabocheDriverState, ChabocheParameterState, ChabocheVariableState

include("DSA.jl")
export DSA, DSADriverState, DSAParameterState, DSAVariableState

# include("viscoplastic.jl")
# export ViscoPlastic

include("uniaxial_increment.jl")
export uniaxial_increment!

end
