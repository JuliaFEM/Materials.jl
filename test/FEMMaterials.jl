module FEMMaterials

using Materials, FEMBase, LinearAlgebra, Tensors

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

material_preprocess_analysis!(material::Material{<:AbstractMaterial}, element::Element{Poi1}, ip, time) = nothing
material_postprocess_analysis!(material::Material{<:AbstractMaterial}, element::Element{Poi1}, ip, time) = nothing
material_preprocess_increment!(material::Material{<:AbstractMaterial}, element::Element{Poi1}, ip, time) = nothing
material_postprocess_increment!(material::Material{<:AbstractMaterial}, element::Element{Poi1}, ip, time) = nothing
material_preprocess_iteration!(material::Material{<:AbstractMaterial}, element::Element{Poi1}, ip, time) = nothing
material_postprocess_iteration!(material::Material{<:AbstractMaterial}, element::Element{Poi1}, ip, time) = nothing

### Chaboche ###

function material_preprocess_analysis!(material::Material{Chaboche}, element, ip, time)
update!(ip, "plastic strain", 0.0 => zeros(6))
update!(ip, "stress", 0.0 => zeros(6))
update!(ip, "strain", 0.0 => zeros(6))
update!(ip, "backstress 1", 0.0 => zeros(6))
update!(ip, "backstress 2", 0.0 => zeros(6))
update!(ip, "cumulative equivalent plastic strain", 0.0 => 0.0)
update!(ip, "R", 0.0 => 0.0)
return nothing
end

function material_preprocess_increment!(material::Material{Chaboche}, element, ip, time)
mat = material.properties
material.dtime = time - material.time
mat.youngs_modulus = element("youngs modulus", ip, time)
mat.poissons_ratio = element("poissons ratio", ip, time)
mat.yield_stress = element("yield stress", ip, time)
mat.K_n = element("K_n", ip, time)
mat.n_n = element("n_n", ip, time)
mat.C_1 = element("C_1", ip, time)
mat.D_1 = element("D_1", ip, time)
mat.C_2 = element("C_2", ip, time)
mat.D_2 = element("D_2", ip, time)
mat.Q = element("Q", ip, time)
mat.b = element("b", ip, time)
return nothing
end

function material_postprocess_increment!(material::Material{Chaboche}, element, ip, time)
# preprocess_increment!(material, element, ip, time)
# integrate_material!(material)
mat = material.properties
material.stress .+= material.dstress
material.strain .+= material.dstrain
material.time += material.dtime
mat.plastic_strain .+= mat.dplastic_strain
mat.cumulative_equivalent_plastic_strain += mat.dcumulative_equivalent_plastic_strain
mat.backstress1 .+= mat.dbackstress1
mat.backstress2 .+= mat.dbackstress2
mat.R += mat.dR
update!(ip, "stress", time => copy(material.stress))
update!(ip, "strain", time => copy(material.strain))
update!(ip, "plastic strain", time => copy(mat.plastic_strain))
update!(ip, "cumulative equivalent plastic strain", time => copy(mat.cumulative_equivalent_plastic_strain))
update!(ip, "backstress 1", time => copy(mat.backstress1))
update!(ip, "backstress 2", time => copy(mat.backstress2))
update!(ip, "R", time => copy(mat.R))
return nothing
end

### IdealPlastic ###

function material_preprocess_increment!(material::Material{IdealPlastic}, element, ip, time)
    material.dtime = time - material.time

    # interpolate / update fields from elements to material
    mat = material.properties
    mat.youngs_modulus = element("youngs modulus", ip, time)
    mat.poissons_ratio = element("poissons ratio", ip, time)

    if haskey(element, "yield stress")
        mat.yield_stress = element("yield stress", ip, time)
    else
        mat.yield_stress = Inf
    end

    if haskey(element, "plastic strain")
        plastic_strain = element("plastic strain", ip, time)
    end

    # reset all incremental variables ready for next iteration
    fill!(mat.dplastic_strain, 0.0)
    mat.dplastic_multiplier = 0.0

    return nothing
end

function material_postprocess_increment!(material::Material{IdealPlastic}, element, ip, time)
    props = material.properties
    # material_preprocess_iteration!(material, element, ip, time)
    # integrate_material!(material) # one more time!
    material.stress += material.dstress
    material.strain += material.dstrain
    material.time += material.dtime
    props.plastic_strain += props.dplastic_strain
    props.plastic_multiplier += props.dplastic_multiplier
    update!(ip, "stress", time => copy(material.stress))
    update!(ip, "strain", time => copy(material.strain))
    return nothing
end


##################
# Viscoplastic JuliaFEM hooks #
##################

function material_preprocess_increment!(material::Material{ViscoPlastic}, element, ip, time)
    material.dtime = time - material.time

    # interpolate / update fields from elements to material
    mat = material.properties
    mat.youngs_modulus = element("youngs modulus", ip, time)
    mat.poissons_ratio = element("poissons ratio", ip, time)

    if haskey(element, "yield stress")
        mat.yield_stress = element("yield stress", ip, time)
    else
        mat.yield_stress = Inf
    end

    if haskey(element, "plastic strain")
        plastic_strain = element("plastic strain", ip, time)
    end

    # reset all incremental variables ready for next iteration
    fill!(mat.dplastic_strain, 0.0)
    mat.dplastic_multiplier = 0.0

    return nothing
end


# """ Material postprocess step after increment finish. """
# function postprocess_increment!(material::Material{M}, element, ip, time) where {M}
#     return nothing
# end

function material_postprocess_increment!(material::Material{ViscoPlastic}, element, ip, time)
    props = material.properties
    # material_preprocess_iteration!(material, element, ip, time)
    # integrate_material!(material) # one more time!
    material.stress += material.dstress
    material.strain += material.dstrain
    material.time += material.dtime
    props.plastic_strain += props.dplastic_strain
    props.plastic_multiplier += props.dplastic_multiplier
    update!(ip, "stress", time => copy(material.stress))
    update!(ip, "strain", time => copy(material.strain))
    return nothing
end


export material_preprocess_analysis!, material_preprocess_increment!,
       material_preprocess_iteration!, material_postprocess_analysis!,
       material_postprocess_increment!, material_postprocess_iteration!

end
