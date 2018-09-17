# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

module Materials

using FEMBase, LinearAlgebra, ForwardDiff, Tensors

abstract type AbstractMaterial end

"""
    Material{M<:AbstractMaterial}

- Stress σ
- Strain ε
- Stress increment Δσ
- Strain increment Δε
- Jacobian matrix of the consitutive model, ∂Δσ/∂Δε

Properties contains internal material state parameters, like Young's modulus,
Poissons ratio, yield stress limit, and so on.

# Example

Let us define linear, isotroopic Hooke material.

struct LinearIsotropicHooke <: AbstractMaterial
    youngs_modulus :: Float64
    poissons_ratio :: Float64
end

"""
mutable struct Material{M<:AbstractMaterial}
    stress :: Vector{Float64}
    strain :: Vector{Float64}
    dstress :: Vector{Float64}
    dstrain :: Vector{Float64}
    jacobian :: Matrix{Float64}
    dtime :: Float64
    properties :: M
end

"""
    Material(M; material_properties...)

Create new material `M`. Any additional properties are passed to material
constructor.

# Example

Assume there's a material `LinearIsotropicHooke`.
"""
function Material(::Type{M}, material_properties) where {M}
    stress = zeros(6)
    strain = zeros(6)
    dstress = zeros(6)
    dstrain = zeros(6)
    jacobian = zeros(6,6)
    dtime = 0.0
    properties = M(material_properties...)
    return Material(stress, strain, dstress, dstrain, jacobian, dtime, properties)
end

export AbstractMaterial, Material

function preprocess_analysis! end
function preprocess_increment! end
function postprocess_analysis! end
function postprocess_increment! end
function integrate_material! end
export preprocess_analysis!
export preprocess_increment!
export postprocess_analysis!
export postprocess_increment!
export integrate_material!

include("idealplastic.jl")
export IdealPlastic

include("simulator.jl")

include("oneelementsimulator.jl")

include("chaboche.jl")
export Chaboche

include("viscoplastic.jl")
export ViscoPlastic

end
