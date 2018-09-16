# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

mutable struct IdealPlastic <: AbstractMaterial
    # Material parameters
    youngs_modulus :: Float64
    poissons_ratio :: Float64
    yield_stress :: Float64
    # Internal state variables
    plastic_strain :: Vector{Float64}
    dplastic_strain :: Vector{Float64}
    plastic_multiplier :: Float64
    dplastic_multiplier :: Float64
    # Other parameters
    use_ad :: Bool
end

function IdealPlastic()
    youngs_modulus = 0.0
    poissons_ratio = 0.0
    yield_stress = 0.0
    plastic_strain = zeros(6)
    dplastic_strain = zeros(6)
    plastic_multiplier = 0.0
    dplastic_multiplier = 0.0
    use_ad = false
    return IdealPlastic(youngs_modulus, poissons_ratio, yield_stress,
                        plastic_strain, dplastic_strain, plastic_multiplier,
                        dplastic_multiplier, use_ad)
end

function FEMBase.initialize!(material::Material{IdealPlastic}, element, ip, time)
    if !haskey(ip, "stress")
        update!(ip, "stress", time => copy(material.stress))
    end
    if !haskey(ip, "strain")
        update!(ip, "strain", time => copy(material.strain))
    end
end

""" Preprocess step of material before analysis start. """
function preprocess_analysis!(material::Material{IdealPlastic}, element, ip, time)

    # TODO
    # for fn in fieldnames(M)
    #     fn2 = replace("$fn", "_" => " ")
    #     if haskey(element, fn2)
    #         fv = element(fn2, ip, time)
    #         setproperty!(mat.properties, fn, fv)
    #     else
    #         warn("Unable to update field $fn from element to material.")
    #     end
    #     if startswith(fn2, "d") && fn2[2:end]
    # end

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

""" Preprocess step before increment start. """
function preprocess_increment!(material::Material{IdealPlastic}, element, ip, time)
    gradu = element("displacement", ip, time, Val{:Grad})
    strain = 0.5*(gradu + gradu')
    strainvec = [strain[1,1], strain[2,2], strain[3,3],
                 2.0*strain[1,2], 2.0*strain[2,3], 2.0*strain[3,1]]
    material.dstrain = strainvec - material.strain
    return nothing
end

""" Material postprocess step after increment finish. """
function postprocess_increment!(material::Material{M}, element, ip, time) where {M}
    return nothing
end

function postprocess_analysis!(material::Material{IdealPlastic})
    props = material.properties
    material.stress .+= material.dstress
    material.strain .+= material.dstrain
    props.plastic_strain .+= props.dplastic_strain
    props.plastic_multiplier += props.dplastic_multiplier
    return nothing
end

function postprocess_analysis!(material::Material{IdealPlastic}, element, ip, time)
    # TODO: why?
    preprocess_increment!(material, element, ip, time)
    integrate_material!(material) # one more time!
    postprocess_analysis!(material)
    update!(ip, "stress", time => copy(material.stress))
    update!(ip, "strain", time => copy(material.strain))
    return nothing
end

function integrate_material!(material::Material{IdealPlastic})
    mat = material.properties

    E = mat.youngs_modulus
    nu = mat.poissons_ratio
    mu = E/(2.0*(1.0+nu))
    lambda = E*nu/((1.0+nu)*(1.0-2.0*nu))

    stress = material.stress
    strain = material.strain
    dstress = material.dstress
    dstrain = material.dstrain
    D = material.jacobian

    fill!(D, 0.0)
    D[1,1] = D[2,2] = D[3,3] = 2.0*mu + lambda
    D[4,4] = D[5,5] = D[6,6] = mu
    D[1,2] = D[2,1] = D[2,3] = D[3,2] = D[1,3] = D[3,1] = lambda

    dstress[:] .= D*dstrain
    stress_tr = stress + dstress
    s11, s22, s33, s12, s23, s31 = stress_tr
    stress_v = sqrt(1/2*((s11-s22)^2 + (s22-s33)^2 + (s33-s11)^2 + 6*(s12^2+s23^2+s31^2)))
    if stress_v <= mat.yield_stress
        fill!(mat.dplastic_strain, 0.0)
        mat.dplastic_multiplier = 0.0
        return nothing
    else
        stress_h = 1.0/3.0*sum(stress_tr[1:3])
        stress_dev = stress_tr - stress_h*[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        n = 3.0/2.0*stress_dev/stress_v
        dla = 1.0/(3.0*mu)*(stress_v - mat.yield_stress)
        dstrain_pl = dla*n
        mat.dplastic_strain = dstrain_pl
        mat.dplastic_multiplier = dla
        dstress[:] .= D*(dstrain - dstrain_pl)
        D[:,:] .-= (D*n*n'*D) / (n'*D*n)
        return nothing
    end

    return nothing

end
