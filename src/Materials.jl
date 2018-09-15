# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

module Materials

using FEMBase, LinearAlgebra

abstract type AbstractMaterial end

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

export AbstractMaterial, Material

# include("olli.jl") # to be uncommented when old code is fixed.

end
