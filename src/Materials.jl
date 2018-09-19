# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

module Materials

using FEMBase, LinearAlgebra, ForwardDiff, Tensors, SparseArrays, NLsolve

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

material_preprocess_increment!(material::Material{<:AbstractMaterial}, element, ip, time) = nothing
material_postprocess_analysis!(material::Material{<:AbstractMaterial}, element, ip, time) = nothing
material_postprocess_increment!(material::Material{<:AbstractMaterial}, element, ip, time) = nothing
material_postprocess_iteration!(material::Material{<:AbstractMaterial}, element, ip, time) = nothing

function material_preprocess_analysis!(material::Material{M}, element, ip, time) where {M<:AbstractMaterial}
    if !haskey(ip, "stress")
        update!(ip, "stress", time => copy(material.stress))
    end
    if !haskey(ip, "strain")
        update!(ip, "strain", time => copy(material.strain))
    end
    material.time = time
    return nothing
end

function material_preprocess_iteration!(material::Material{M}, element, ip, time) where {M<:AbstractMaterial}
    gradu = element("displacement", ip, time, Val{:Grad})
    strain = 0.5*(gradu + gradu')
    strainvec = [strain[1,1], strain[2,2], strain[3,3],
                 2.0*strain[1,2], 2.0*strain[2,3], 2.0*strain[3,1]]
    material.dstrain = strainvec - material.strain
    return nothing
end

export material_preprocess_analysis!, material_preprocess_increment!,
       material_preprocess_iteration!, material_postprocess_analysis!,
       material_postprocess_increment!, material_postprocess_iteration!

export integrate_material!

include("idealplastic.jl")
export IdealPlastic

# Material point simulator to study material behavior in single integration point
include("simulator.jl")

include("simulator2.jl")
export uniaxial_increment!

include("chaboche.jl")
export Chaboche

# Material simulator to solve global system and run standard one element tests
include("mecamatso.jl")
export get_one_element_material_analysis, AxialStrainLoading, ShearStrainLoading, update_bc_elements!

include("viscoplastic.jl")
export ViscoPlastic

end
