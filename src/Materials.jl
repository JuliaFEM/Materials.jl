# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

module Materials

using LinearAlgebra, ForwardDiff, Tensors, NLsolve

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
    time :: Float64
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
    time = 0.0
    dtime = 0.0
    properties = M(material_properties...)
    return Material(stress, strain, dstress, dstrain, jacobian, time, dtime, properties)
end

export AbstractMaterial, Material

function integrate_material!(material::Material{M}) where {M<:AbstractMaterial}
    error("One needs to define how to integrate material $M!")
end

export integrate_material!

include("idealplastic.jl")
export IdealPlastic

include("chaboche.jl")
export Chaboche

include("viscoplastic.jl")
export ViscoPlastic

include("uniaxial_increment.jl")
export uniaxial_increment!

end
