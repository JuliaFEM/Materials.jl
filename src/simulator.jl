# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Materials

mutable struct Simulator
    stresses :: Vector{Vector{Float64}}
    strains :: Vector{Vector{Float64}}
    times :: Vector{Float64}
    material :: Material{<:AbstractMaterial}
end

function Simulator(material)
    return Simulator([],[],[],material)
end

function initialize!(simulator, strains, times)
    simulator.strains = strains
    simulator.times = times
    return nothing
end

function run!(simulator)
    material = simulator.material
    times = simulator.times
    strains = simulator.strains
    t_n = times[1]
    strain_n = strains[1]
    push!(simulator.stresses, copy(material.stress))
    for i in 2:length(times)
        strain = strains[i]
        t = times[i]
        dstrain = strain - strain_n
        dt = t - t_n
        material.dstrain = dstrain
        material.dtime = dt
        integrate_material!(material)
        material_postprocess_increment!(material)
        push!(simulator.stresses, copy(material.stress))
        strain_n = strain
        t_n = t
    end
    return nothing
end
