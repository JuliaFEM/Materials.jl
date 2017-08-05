# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

abstract type AbstractMaterial end

abstract type Elastic<:AbstractMaterial end
abstract type Plastic<:AbstractMaterial end

# Elastic models
struct IsotropicHooke<:Elastic
    youngs_modulus :: AbstractFloat
    nu :: AbstractFloat
end

# Plastic models
type NoPlasticity <: Plastic end

struct VonMises{F <: AbstractFloat} <: Plastic
    yield_stress :: F
end

struct Model
    elastic :: Elastic
    plastic :: Plastic
end

Model(x::Elastic) = Model(x, NoPlasticity())

struct Material
    dimension :: Int
    finite_strain :: Bool
    formulation :: Symbol
    history_values :: Dict{AbstractString, Array{Any}}
    model :: Model
    time :: Vector{AbstractFloat}
    trial_values :: Dict{AbstractString, Any}
end

function create_material(dim, model; formulation=:basic, finite_strain=false)
    time_array = Vector{AbstractFloat}(0)
    trial_values = Dict{AbstractString, Any}()
    history_values = Dict{AbstractString, Array{Any}}()
    history_values["stress"] = [Tensor{2,3}(zeros(3,3))]
    history_values["strain"] = [Tensor{2,3}(zeros(3,3))]
    Material(dim,
             finite_strain,
             formulation,
             history_values,
             model,
             time_array,
             trial_values,
             )
end
