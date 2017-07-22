# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

#type Elasticity <: FieldProblem
#    # these are found from problem.properties for type Problem{Elasticity}
#    formulation :: Symbol
#    finite_strain :: Bool
#    geometric_stiffness :: Bool
#    store_fields :: Vector{Symbol}
#end

abstract type AbstractMaterial end

abstract type Elastic<:AbstractMaterial end

type IsotropicHooke<:Elastic
    youngs_modulus :: AbstractFloat
    nu :: AbstractFloat
end

abstract type Plastic<:AbstractMaterial end

abstract type HyperElastic<:AbstractMaterial end

type Material{P<:AbstractMaterial}
    dimension :: Int
    formulation :: Symbol
    finite_strain :: Bool
    time_steps :: Vector{AbstractFloat}
    properties :: Dict{AbstractString, P}
end

Material(dim; formulation=:test, finite_strain=false) = Material(dim,
                                                                 formulation,
                                                                 finite_strain,
                                                                 Vector{AbstractFloat}(),
                                                                 Dict{AbstractString, AbstractMaterial}())

function add_property!(material::Material, mat_property, name, params...)
    material.properties[name] = mat_property(params...)
end

function add_property!{P<:AbstractMaterial}(material::Material, mat_property::P, name)
    material.properties[name] = mat_property
end

mat = Material(1)