# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

mbv = MFrontInterface.behaviour

mutable struct MFront <: AbstractMaterial
    behaviour::BehaviourAllocated
    behaviour_data::BehaviourDataAllocated
    offset::Integer
    equivalent_plastic_strain::AbstractFloat

end

function MFront_Norton()
    behaviour = load("data/libBehaviour.so","Norton", mbv.Tridimensional)
    behaviour_data = BehaviourData(behaviour)

    offset = get_variable_offset(get_internal_state_variables(behaviour),
                                 "EquivalentViscoplasticStrain",
                                 get_hypothesis(behaviour))

    set_external_state_variable!(get_final_state(behaviour_data), "Temperature", 293.15)
    update(behaviour_data)

    return MFront(behaviour, behaviour_data, offset, 0.0)
end

function integrate_material!(material::Material{MFront})
    mat = material.properties
    behaviour = mat.behaviour
    behaviour_data = mat.behaviour_data
    offset = mat.offset
    eps = mat.equivalent_plastic_strain

    stress = material.stress
    strain = material.strain
    dstress = material.dstress
    dstrain = material.dstrain
    jacobian = material.jacobian
    time = material.time
    dtime = material.dtime

    set_time_increment!(behaviour_data, dtime)
    update(behaviour_data)

    # Maybe use LibrariesManager::getInternalStateVariablesNames

    gradients = get_gradients(get_final_state(behaviour_data))
    gradients[1] += dstrain[1]
    gradients[2] += dstrain[2]
    gradients[3] += dstrain[3]
    gradients[4] += dstrain[4]
    gradients[5] += dstrain[5]
    gradients[6] += dstrain[6]

    integrate(behaviour_data, behaviour)
    update(behaviour_data)

    eps = get_internal_state_variables(get_final_state(behaviour_data))[offset]
end
