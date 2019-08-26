# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

mutable struct MFront <: AbstractMaterial

    mbv = MFrontInterface.behaviour
    behaviour = load("data/libBehaviour.so","Norton", mbv.Tridimensional)
    behaviour_data = BehaviourData(behaviour)
    set_time_increment!(behaviour_data, dtime)

    offset = get_variable_offset(get_internal_state_variables(behaviour),
                                "EquivalentViscoplasticStrain",
                                get_hypothesis(behaviour))

    set_external_state_variable!(get_final_state(behaviour_data), "Temperature", 293.15)

    update(behaviour_data)

    return MFront(behaviour, behaviour_data, offset, )
end

function integrate_material!(material::Material{MFront})
    mat = material.properties
    behaviour = mat.behaviour
    behaviour_data = mat.behaviour_data
    offset = mat.offset

    stress = material.stress
    strain = material.strain
    dstress = material.dstress
    dstrain = material.dstrain
    jacobian = material.jacobian
    time = material.time
    dtime = material.dtime

    # Maybe use LibrariesManager::getInternalStateVariablesNames

    integrate(behaviour_data, behaviour)
    update(behaviour_data)
    get_gradients(get_final_state(behaviour_data))[1] += de
    push!(p,get_internal_state_variables(get_final_state(behaviour_data))[offset])
end
