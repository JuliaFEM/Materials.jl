# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

module Materials

using LinearAlgebra, ForwardDiff, Tensors, NLsolve, Parameters

abstract type AbstractMaterial end
abstract type AbstractMaterialState end

"""
    :+(state::T, dstate::T) where {T <: AbstractMaterialState}

Addition for material states.

Given two material states `state` and `dstate` of type `T`, add each field of
`dstate` into the corresponding field of `state`. Return the resulting material
state.
"""
@generated function Base.:+(state::T, dstate::T) where {T <: AbstractMaterialState}
   expr = [:(state.$p + dstate.$p) for p in fieldnames(T)]
   return :(T($(expr...)))
end

export AbstractMaterial, AbstractMaterialState

"""
    integrate_material!(material::M) where {M<:AbstractMaterial}

Integrate one timestep. The input `material` represents the problem state at the
end of the previous timestep.

Abstract method. Must be implemented for each material type. When integration is
done, the method must update the `material` argument to have the new state.
"""
function integrate_material!(material::M) where {M<:AbstractMaterial}
    error("One needs to define how to integrate material $M!")
end

"""
    update_material!(material::M) where {M <: AbstractMaterial}

In `material`, add `ddrivers` into `drivers`, `dparameters` into `parameters`,
and replace `variables` by `variables_new`. Then `reset_material!`.
"""
function update_material!(material::M) where {M <: AbstractMaterial}
    material.drivers += material.ddrivers
    material.parameters += material.dparameters
    material.variables = material.variables_new
    reset_material!(material)
    return nothing
end

"""
    reset_material!(material::M) where {M <: AbstractMaterial}

In `material`, zero out `ddrivers`, `dparameters` and `variables_new`.

Used internally by `update_material!`.
"""
function reset_material!(material::M) where {M <: AbstractMaterial}
    material.ddrivers = typeof(material.ddrivers)()
    material.dparameters = typeof(material.dparameters)()
    material.variables_new = typeof(material.variables_new)()
    return nothing
end

export integrate_material!, update_material!, reset_material!

include("utilities.jl")
export Symm2, Symm4, lame, delame, isotropic_elasticity_tensor, debang

include("idealplastic.jl")
export IdealPlastic, IdealPlasticDriverState, IdealPlasticParameterState, IdealPlasticVariableState

include("chaboche.jl")
export Chaboche, ChabocheDriverState, ChabocheParameterState, ChabocheVariableState

# include("viscoplastic.jl")
# export ViscoPlastic

include("uniaxial_increment.jl")
export uniaxial_increment!

include("biaxial_increment.jl")
export biaxial_increment!

include("stress_driven_uniaxial_increment.jl")
export stress_driven_uniaxial_increment!

end
