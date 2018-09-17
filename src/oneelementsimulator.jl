# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using JuliaFEM, FEMBase, LinearAlgebra, Materials

include("../examples/continuum.jl")
abstract type AbstractLoading end
mutable struct OneElementSimulator
    stresses :: Vector{Vector{Float64}}
    strains :: Vector{Vector{Float64}}
    times :: Vector{Float64}
    material :: Material{<:AbstractMaterial}
    elements :: Vector{Element}
    bc_elements :: Vector{Element}
    loading :: AbstractLoading
end

mutable struct AxialStrainLoading <: AbstractLoading
    times :: Vector{Float64}
    loads :: Vector{Float64}
end
mutable struct ShearStrainLoading <: AbstractLoading
    times :: Vector{Float64}
    loads :: Vector{Float64}
end

function OneElementSimulator(material, loading::AbstractLoading)
    X = Dict(
        1 => [0.0, 0.0, 0.0],
        2 => [1.0, 0.0, 0.0],
        3 => [1.0, 1.0, 0.0],
        4 => [0.0, 1.0, 0.0],
        5 => [0.0, 0.0, 1.0],
        6 => [1.0, 0.0, 1.0],
        7 => [1.0, 1.0, 1.0],
        8 => [0.0, 1.0, 1.0])
    element = Element(Hex8, (1, 2, 3, 4, 5, 6, 7, 8))
    elements = [element]
    update!(elements, "geometry", X)
    # Initialize integration points for internal variable storage
    for element in elements
        for ip in get_integration_points(element)
            FEMBase.initialize!(material, element, ip, 0.0)
            ip.fields["material"] = field(material)
        end
    end
    # Get bc_elements
    bc_elements = get_bc_elements(loading)
    update!(bc_elements, "geometry", X)
    return OneElementSimulator([],[],[], material, elements, bc_elements, loading)
end

function get_bc_elements(loading::AxialStrainLoading)
    bc_element_1 = Element(Poi1, (1,))
    bc_element_2 = Element(Poi1, (2,))
    bc_element_3 = Element(Poi1, (3,))
    bc_element_4 = Element(Poi1, (4,))
    bc_element_5 = Element(Poi1, (5,))
    bc_element_6 = Element(Poi1, (6,))
    bc_element_7 = Element(Poi1, (7,))
    bc_element_8 = Element(Poi1, (8,))
    bc_elements = [bc_element_1, bc_element_2, bc_element_3, bc_element_4,
                   bc_element_5, bc_element_6, bc_element_7, bc_element_8]
    # Fix bottom side
    for element in (bc_element_1, bc_element_2, bc_element_3, bc_element_4)
        update!(element, "displacement 3", 0.0)
    end
    # Set loading bc
    for element in (bc_element_5, bc_element_6, bc_element_7, bc_element_8)
        for (t,f) in zip(loading.times, loading.loads)
            # @info "Setting displacement 3 at $(t) to $(f)!"
            update!(element, "displacement 3", t => f)
        end
    end
    # Set rest of boundary conditions
    update!(bc_element_1, "displacement 1", 0.0)
    update!(bc_element_1, "displacement 2", 0.0)
    update!(bc_element_2, "displacement 2", 0.0)
    update!(bc_element_4, "displacement 1", 0.0)

    update!(bc_element_5, "displacement 1", 0.0)
    update!(bc_element_5, "displacement 2", 0.0)
    update!(bc_element_6, "displacement 2", 0.0)
    update!(bc_element_8, "displacement 1", 0.0)
    return bc_elements
end

function get_bc_elements(loading::ShearStrainLoading)
    bc_element_1 = Element(Poi1, (1,))
    bc_element_2 = Element(Poi1, (2,))
    bc_element_3 = Element(Poi1, (3,))
    bc_element_4 = Element(Poi1, (4,))
    bc_element_5 = Element(Poi1, (5,))
    bc_element_6 = Element(Poi1, (6,))
    bc_element_7 = Element(Poi1, (7,))
    bc_element_8 = Element(Poi1, (8,))
    bc_elements = [bc_element_1, bc_element_2, bc_element_3, bc_element_4,
                   bc_element_5, bc_element_6, bc_element_7, bc_element_8]
    # Fix bottom side
    for element in (bc_element_1, bc_element_2, bc_element_3, bc_element_4)
        update!(element, "displacement 1", 0.0)
        update!(element, "displacement 2", 0.0)
        update!(element, "displacement 3", 0.0)
    end
    # Set top side bcs
    for element in (bc_element_5, bc_element_6, bc_element_7, bc_element_8)
        for (t,f) in zip(loading.times, loading.loads)
            update!(element, "displacement 1", t => f)
        end
        update!(element, "displacement 2", 0.0)
        update!(element, "displacement 3", 0.0)
    end
    return bc_elements
end

function update_simulator_state!(simulator, time)
    ip = first(get_integration_points(first(simulator.elements)))
    stress = ip("stress", time)
    strain = ip("strain", time)
    @info("time = $time, stress = $stress, strain = $strain")
    push!(simulator.stresses, copy(stress))
    push!(simulator.strains, copy(strain))
    push!(simulator.times, time)
    return
end

function initialize!(simulator)
    update_simulator_state!(simulator, 0.0)
    return nothing
end

function run!(simulator::OneElementSimulator)
    material = simulator.material
    times = simulator.loading.times
    body = Problem(Continuum3D, "1 element problem", 3)
    bc = Problem(Dirichlet, "fix displacement", 3, "displacement")
    add_elements!(body, simulator.elements)
    add_elements!(bc, simulator.bc_elements)
    analysis = Analysis(Nonlinear, "solve problem")
    add_problems!(analysis, body, bc)
    for problem in get_problems(analysis)
        FEMBase.initialize!(problem, analysis.properties.time)
    end
    update!(simulator.elements, "displacement", analysis.properties.time => Dict(j => zeros(3) for j in 1:8))
    for i in 2:length(times)
        dtime = times[i] - times[i-1]
        material.dtime = dtime
        analysis.properties.time += dtime
        @info("time = $(analysis.properties.time)")
        for element in simulator.elements
            for ip in get_integration_points(element)
                material = ip("material", analysis.properties.time)
                material.dtime = dtime
                preprocess_analysis!(material, element, ip, analysis.properties.time)
            end
        end
        FEMBase.run!(analysis)
        for element in simulator.elements
            for ip in get_integration_points(element)
                material = ip("material", analysis.properties.time)
                postprocess_analysis!(material, element, ip, analysis.properties.time)
            end
        end
        update_simulator_state!(simulator, analysis.properties.time)
        # update simulator stress and strain
    end
    return nothing
end
