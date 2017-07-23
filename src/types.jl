# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract type AbstractMaterial end

abstract type Elastic<:AbstractMaterial end
abstract type Plastic<:AbstractMaterial end
abstract type HyperElastic<:AbstractMaterial end

type IsotropicHooke<:Elastic
    youngs_modulus :: AbstractFloat
    nu :: AbstractFloat
end

type VonMises<:Plastic
    yield_stress :: AbstractFloat
    yield_function :: Function
end

type Material{P<:AbstractMaterial}
    dimension :: Int
    formulation :: Symbol
    finite_strain :: Bool
    time :: Vector{AbstractFloat}
    properties :: Dict{AbstractString, P}
    trial_values :: Dict{AbstractString, Any}
    history_values :: Dict{AbstractString, Array{Any}}
end

Material(dim; formulation=:test, finite_strain=false) = Material(dim,
                                                                 formulation,
                                                                 finite_strain,
                                                                 Vector{AbstractFloat}(0),
                                                                 Dict{AbstractString, AbstractMaterial}(),
                                                                 Dict{AbstractString, Any}(),
                                                                 Dict{AbstractString, Array{Any}}()
                                                                 )

function add_property!(material::Material, mat_property, name, params...)
    material.properties[name] = mat_property(params...)
end

function add_property!{P<:AbstractMaterial}(material::Material, mat_property::P, name)
    material.properties[name] = mat_property
end

mat = Material(1)