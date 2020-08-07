# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

module Materials

abstract type AbstractMaterial end
abstract type AbstractMaterialState end

export AbstractMaterial, AbstractMaterialState
export integrate_material!, update_material!, reset_material!

"""
    :+(state::T, dstate::T) where T <: AbstractMaterialState

Addition for material states.

Given two material states `state` and `dstate` of type `T`, add each field of
`dstate` into the corresponding field of `state`. Return the resulting material
state.
"""
@generated function Base.:+(state::T, dstate::T) where T <: AbstractMaterialState
   expr = [:(state.$p + dstate.$p) for p in fieldnames(T)]
   return :(T($(expr...)))
end

"""
    integrate_material!(material::AbstractMaterial)

Integrate one timestep. The input `material.variables` represents the old
problem state.

Abstract method. Must be implemented for each material type. When integration is
done, the method **must** write the new state into `material.variables_new`.

**Do not** write into `material.variables`; actually committing the timestep
(i.e. accepting that one step of time evolution and applying it permanently)
is the job of `update_material!`.
"""
function integrate_material!(material::M) where M <: AbstractMaterial
    error("One needs to define how to integrate material $M!")
end

"""
    update_material!(material::AbstractMaterial)

Commit the result of `integrate_material!`.

In `material`, we add `ddrivers` into `drivers`, `dparameters` into
`parameters`, and replace `variables` by `variables_new`. Then we
automatically invoke `reset_material!`.
"""
function update_material!(material::AbstractMaterial)
    material.drivers += material.ddrivers
    material.parameters += material.dparameters
    material.variables = material.variables_new
    reset_material!(material)
    return nothing
end

"""
    reset_material!(material::AbstractMaterial)

In `material`, we zero out `ddrivers`, `dparameters` and `variables_new`. This
clears out the tentative state produced when a timestep has been computed, but
has not yet been committed.

Used internally by `update_material!`.
"""
function reset_material!(material::AbstractMaterial)
    material.ddrivers = typeof(material.ddrivers)()
    material.dparameters = typeof(material.dparameters)()
    material.variables_new = typeof(material.variables_new)()
    return nothing
end

include("utilities.jl")
using .Utilities
export Symm2, Symm4
export delta, II, IT, IS, IA, IV, ID, isotropic_elasticity_tensor, isotropic_compliance_tensor
export lame, delame, debang, find_root

include("perfectplastic.jl")
using .PerfectPlasticModule
export PerfectPlastic, PerfectPlasticDriverState, PerfectPlasticParameterState, PerfectPlasticVariableState

include("chaboche.jl")
using .ChabocheModule
export Chaboche, ChabocheDriverState, ChabocheParameterState, ChabocheVariableState

include("memory.jl")
using .MemoryModule
export Memory, MemoryDriverState, MemoryParameterState, MemoryVariableState

include("DSA.jl")
using .DSAModule
export DSA, DSADriverState, DSAParameterState, DSAVariableState

include("increments.jl")
using .Increments
export uniaxial_increment!, biaxial_increment!, stress_driven_uniaxial_increment!

end
