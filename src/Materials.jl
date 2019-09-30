# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

module Materials

using LinearAlgebra, ForwardDiff, Tensors, NLsolve, Parameters

abstract type AbstractMaterial end
abstract type AbstractMaterialState end

@generated function Base.:+(state::T, dstate::T) where {T <: AbstractMaterialState}
   expr = [:(state.$p+ dstate.$p) for p in fieldnames(T)]
   return :(T($(expr...)))
end

export AbstractMaterial

function integrate_material!(material::M) where {M<:AbstractMaterial}
    error("One needs to define how to integrate material $M!")
end

function update_material!(material::M) where {M <: AbstractMaterial}
    material.drivers += material.ddrivers
    material.parameters += material.dparameters
    material.variables = material.variables_new
    reset_material!(material)
end

function reset_material!(material::M) where {M <: AbstractMaterial}
    material.ddrivers = typeof(material.ddrivers)()
    material.dparameters = typeof(material.dparameters)()
    material.variables_new = typeof(material.variables_new)()
end

export integrate_material!, update_material!, reset_material!

include("idealplastic.jl")
export IdealPlastic, IdealPlasticDriverState, IdealPlasticParameterState, IdealPlasticVariableState

include("chaboche.jl")
export Chaboche, ChabocheDriverState, ChabocheParameterState, ChabocheVariableState

# include("viscoplastic.jl")
# export ViscoPlastic

include("uniaxial_increment.jl")
export uniaxial_increment!

end
