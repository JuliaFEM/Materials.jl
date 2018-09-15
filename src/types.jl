# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md
using FEMBase, LinearAlgebra

abstract type AbstractMaterial end
abstract type Elastic<:AbstractMaterial end
abstract type Plastic<:AbstractMaterial end

struct NoPlasticity <: Plastic
    i
end

NoPlasticity() = NoPlasticity(0)

"""
    Material{M<:AbstractMaterial}

Stress σ, strain ε, material_stiffness D = Δσ/Δε, D(t) = ∂σ(t)/∂ε(t).
Properties contains internal material state parameters.

# Example

Let us define linear, isotroopic Hooke material.

struct LinearIsotropicHooke <: AbstractMaterial
    youngs_modulus :: Float64
    poissons_ratio :: Float64
end

"""
struct Material{M<:AbstractMaterial}
    stress :: Vector{Float64}
    strain :: Vector{Float64}
    material_stiffness :: Matrix{Float64}
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
    material_stiffness = zeros(6,6)
    @info("material properties = $material_properties")
    properties = M(material_properties...)
    return Material(stress, strain, material_stiffness, properties)
end


# Elastic models
struct IsotropicHooke<:Elastic
    youngs_modulus :: AbstractFloat
    nu :: AbstractFloat
end

struct VonMises{F <: AbstractFloat} <: Plastic
    yield_stress :: F
end

struct Model
    elastic :: Elastic
    plastic :: Plastic
end

Model(x::Elastic) = Model(x, NoPlasticity())

export AbstractMaterial, Material