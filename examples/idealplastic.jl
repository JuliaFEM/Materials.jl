# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

abstract type AbstractMaterial end

mutable struct IdealPlastic <: AbstractMaterial
    # Material parameters
    youngs_modulus :: Float64
    poissons_ratio :: Float64
    yield_stress :: Float64
    # Internal state variables
    plastic_strain :: Matrix{Float64}
    dplastic_strain :: Matrix{Float64}
    plastic_multiplier :: Float64
    dplastic_multiplier :: Float64
end

function IdealPlastic(element, ip, time)
    # Material parameters
    youngs_modulus = element("youngs modulus", ip, time)
    poissons_ratio = element("poissons ratio", ip, time)
    yield_stress = element("yield stress", ip, time)
    # Internal variables
    plastic_strain = element("plastic strain", ip, time)
    dplastic_strain = zeros(3, 3)
    plastic_multiplier = 0.0
    dplastic_multiplier = 0.0
    return IdealPlastic(youngs_modulus, poissons_ratio, yield_stress,
                        plastic_strain, dplastic_strain, plastic_multiplier,
                        dplastic_multiplier)
end

function calculate_stress!(material::AbstractMaterial, element, ip, time, dtime,
                           material_matrix, stress_vector)
    # Update material parameters
    material.youngs_modulus = element("youngs modulus", ip, time)
    material.poissons_ratio = element("poissons ratio", ip, time)
    material.yield_stress = element("yield stress", ip, time)

    gradu0 = element("displacement", ip, time-dtime, Val{:Grad})
    gradu = element("displacement", ip, time, Val{:Grad})
    X = element("geometry", ip, time)

    strain0 = 0.5*(gradu0 + gradu0')
    strain = 0.5*(gradu + gradu')
    dstrain = strain - strain0

    E = material.youngs_modulus
    nu = material.poissons_ratio
    mu = E/(2.0*(1.0+nu))
    lambda = E*nu/((1.0+nu)*(1.0-2.0*nu))
    G = 0.5*E/(1.0+nu)

    strain_elastic0 = strain0 - material.plastic_strain
    stress0 = lambda*tr(strain_elastic0)*I + 2.0*mu*strain_elastic0

    strain_elastic = strain - material.plastic_strain
    stress_trial = lambda*tr(strain_elastic)*I + 2.0*mu*strain_elastic

    stress_dev = stress_trial - 1.0/3.0*tr(stress_trial)*I
    stress_v = sqrt(3/2*sum(stress_dev .* stress_dev))
    update!(ip, "stress_v", time => stress_v)
    #@info("stuff", stress_v, material.yield_stress)

    fill!(material_matrix, 0.0)
    material_matrix[1,1] = 2.0*mu + lambda
    material_matrix[2,2] = 2.0*mu + lambda
    material_matrix[3,3] = 2.0*mu + lambda
    material_matrix[4,4] = mu
    material_matrix[5,5] = mu
    material_matrix[6,6] = mu
    material_matrix[1,2] = lambda
    material_matrix[2,1] = lambda
    material_matrix[2,3] = lambda
    material_matrix[3,2] = lambda
    material_matrix[1,3] = lambda
    material_matrix[3,1] = lambda

    # if stress_v > 150.0
    #     @info("stress_v = $stress_v")
    #     @info("stress_trial = $stress_trial")
    #     error("this should not be happening")
    # end

    if stress_v < material.yield_stress
        stress_vector[1] = stress_trial[1,1]
        stress_vector[2] = stress_trial[2,2]
        stress_vector[3] = stress_trial[3,3]
        stress_vector[4] = stress_trial[1,2]
        stress_vector[5] = stress_trial[2,3]
        stress_vector[6] = stress_trial[3,1]
        return nothing
    else
        # @info "Plastic strain at X = $X, element # $(element.id), time = $time, stress_v = $stress_v"
        # error("should not")
        # error("plasticity on time $time")
        n = 3.0/2.0*stress_dev/stress_v
        dla = (stress_v - material.yield_stress)/(3.0*G)
        dstrain_pl = dla*n
        material.dplastic_strain = dstrain_pl
        material.dplastic_multiplier = dla
        dstrain_el = dstrain - dstrain_pl
        dstress = lambda*tr(dstrain_el)*I + 2.0*mu*dstrain_el
        stress = stress0 + dstress
        stress_vector[1] = stress[1,1]
        stress_vector[2] = stress[2,2]
        stress_vector[3] = stress[3,3]
        stress_vector[4] = stress[1,2]
        stress_vector[5] = stress[2,3]
        stress_vector[6] = stress[3,1]
        D = material_matrix
        dg = df = [n[1,1], n[2,2], n[3,3], n[1,2], n[2,3], n[3,1]]
        material_matrix[:,:] .= D - (D*dg*df'*D) / (df'*D*dg)
        material_matrix[abs.(material_matrix) .< 1.0e-9] .= 0.0
        #@info("results", material_matrix, stress0, dstrain)
    end

    return nothing
end
