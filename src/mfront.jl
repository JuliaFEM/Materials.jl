using Parameters, MFrontInterface

mgis_bv = MFrontInterface.behaviour

"""
Variables updated by MFront.
"""
@with_kw struct MFrontVariableState <: AbstractMaterialState
    jacobian :: Array{Float64,2} = zeros(Float64, 6, 6)
    stress :: Array{Float64,1} = zeros(Float64, 6)
end

"""
Variables passed in for information.
These drive evolution of the material state.
"""
@with_kw struct MFrontDriverState <: AbstractMaterialState
    strain :: Array{Float64,1} = zeros(Float64, 6)
    time :: Float64 = 0.0
end

"""
Material external variables in order that is specific to chosen MFront behaviour.
"""
@with_kw struct MFrontExternalVariableState <: AbstractMaterialState
    names :: Array{String,1} = [""]
    values :: Array{Float64,1} = zeros(Float64, 1)
end

"""
MFront material structure.

`lib_path` is the path to the compiled shared library.
`behaviour` is the material behaviour name to call from the shared library.
    Generally it's the name of .mfront file, e.g. `Elasticity`.
`hypothesis` is the modelling hypothesis to be used, e.g. `MFrontInterface.behaviour.Tridimensional`
"""
@with_kw mutable struct MFrontMaterial <: AbstractMaterial
    drivers :: MFrontDriverState = MFrontDriverState()
    ddrivers :: MFrontDriverState = MFrontDriverState()
    variables :: MFrontVariableState = MFrontVariableState()
    variables_new :: MFrontVariableState = MFrontVariableState()

    external_variables :: MFrontExternalVariableState = MFrontExternalVariableState()
    
    behaviour :: MFrontInterface.behaviour.BehaviourAllocated
    behaviour_data :: MFrontInterface.behaviour.BehaviourDataAllocated
end

function integrate_material!(material::MFrontMaterial)
    behaviour = material.behaviour
    behaviour_data = material.behaviour_data

    mgis_bv.set_time_increment!(behaviour_data, material.ddrivers.time)

    # setting the external variables (like temperature)
    for j in 1:length(material.external_variables.names)
        if material.external_variables.names[j] != ""
            mgis_bv.set_external_state_variable!(mgis_bv.get_final_state(behaviour_data), material.external_variables.names[j], material.external_variables.values[j])
        end
    end

    # passing strain from material struct to the mfront interface
    dstrain = material.ddrivers.strain
    gradients = mgis_bv.get_gradients(mgis_bv.get_final_state(behaviour_data))
    for j in 1:6
        gradients[j] += dstrain[j]
    end

    # tell mfront interface to calculate the tangent
    # if K[0] is greater than 3.5, the consistent tangent operator must be computed.
    dummy = zeros(36)
    dummy[1] = 4.0
    mgis_bv.set_tangent_operator!(behaviour_data, dummy)

    mgis_bv.integrate(behaviour_data, behaviour)

    stress = [mgis_bv.get_thermodynamic_forces(mgis_bv.get_final_state(behaviour_data))[k] for k in 1:6]
    jacobian = reshape([mgis_bv.get_tangent_operator(behaviour_data)[k] for k in 1:36], 6, 6)
    variables_new = MFrontVariableState(stress=stress, jacobian=jacobian)
    material.variables_new = variables_new
end

function update_material!(material::MFrontMaterial)
    mgis_bv.update(material.behaviour_data)

    material.drivers += material.ddrivers
    material.variables = material.variables_new

    reset_material!(material)
end

function reset_material!(material::MFrontMaterial)
    material.ddrivers = typeof(material.ddrivers)()
    material.variables_new = typeof(material.variables_new)()
end
