# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

function deviator(stress::Vector{Float64})
    return stress - 1.0/3.0*sum(stress[1:3])*[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
end

function von_mises_stress(stress::Vector{Float64})
    return sqrt(0.5*((stress[1]-stress[2])^2 + (stress[2]-stress[3])^2 +
        (stress[3]-stress[1])^2 + 6*(stress[4]^2+stress[5]^2+stress[6]^2)))
end

mutable struct Chaboche <: AbstractMaterial
    # Material parameters
    youngs_modulus :: Float64
    poissons_ratio :: Float64
    yield_stress :: Float64
    K_n :: Float64
    n_n :: Float64
    C_1 :: Float64
    D_1 :: Float64
    C_2 :: Float64
    D_2 :: Float64
    Q :: Float64
    b :: Float64
    # Internal state variables
    plastic_strain :: Vector{Float64}
    dplastic_strain :: Vector{Float64}
    cumulative_equivalent_plastic_strain :: Float64
    dcumulative_equivalent_plastic_strain :: Float64
    backstress1 :: Vector{Float64}
    dbackstress1 :: Vector{Float64}
    backstress2 :: Vector{Float64}
    dbackstress2 :: Vector{Float64}
    R :: Float64
    dR :: Float64
end

function Chaboche()
    youngs_modulus = 0.0
    poissons_ratio = 0.0
    yield_stress = 0.0
    K_n = 0.0
    n_n = 0.0
    C_1 = 0.0
    D_1 = 0.0
    C_2 = 0.0
    D_2 = 0.0
    Q = 0.0
    b = 0.0
    # Internal state variables
    plastic_strain = zeros(6)
    dplastic_strain = zeros(6)
    cumulative_equivalent_plastic_strain = 0.0
    dcumulative_equivalent_plastic_strain = 0.0
    backstress1 = zeros(6)
    dbackstress1 = zeros(6)
    backstress2 = zeros(6)
    dbackstress2 = zeros(6)
    R = 0.0
    dR = 0.0
    return Chaboche(youngs_modulus, poissons_ratio, yield_stress, K_n, n_n, C_1, D_1, C_2, D_2,
                    Q, b, plastic_strain, dplastic_strain, cumulative_equivalent_plastic_strain,
                    dcumulative_equivalent_plastic_strain, backstress1, dbackstress1,
                    backstress2, dbackstress2, R, dR)
end

function integrate_material!(material::Material{Chaboche})
    mat = material.properties
    E = mat.youngs_modulus
    nu = mat.poissons_ratio
    mu = E/(2.0*(1.0+nu))
    lambda = E*nu/((1.0+nu)*(1.0-2.0*nu))
    R0 = mat.yield_stress

    K_n = mat.K_n
    n_n = mat.n_n

    stress = material.stress
    strain = material.strain
    dstress = material.dstress
    dstrain = material.dstrain
    D = material.jacobian

    dplastic_strain = mat.dplastic_strain
    dcumulative_equivalent_plastic_strain = mat.dcumulative_equivalent_plastic_strain
    dbackstress1 = mat.dbackstress1
    dbackstress2 = mat.dbackstress2
    dR = mat.dR

    # peeq = material.cumulative_equivalent_plastic_strain
    X_1 = mat.backstress1
    X_2 = mat.backstress2
    R = mat.R

    fill!(D, 0.0)
    D[1,1] = D[2,2] = D[3,3] = 2.0*mu + lambda
    D[4,4] = D[5,5] = D[6,6] = mu
    D[1,2] = D[2,1] = D[2,3] = D[3,2] = D[1,3] = D[3,1] = lambda

    dstress[:] .= D*dstrain
    stress_tr = stress + dstress

    f_tr = von_mises_stress(stress_tr - X_1 - X_2) - (R0 + R)
    if f_tr <= 0.0
        fill!(dplastic_strain, 0.0)
        mat.dcumulative_equivalent_plastic_strain = 0.0
        fill!(dbackstress1, 0.0)
        fill!(dbackstress2, 0.0)
        mat.dR = 0.0
        return nothing
    else
        # R_n = copy(R)
        # X_1n = copy(X_1)
        # X_2n = copy(X_2)
        # stress_n = copy(stress)
        g! = create_nonlinear_system_of_equations(material, dstrain, material.dtime)
        x0 = [stress_tr; R; X_1; X_2]
        F = similar(x0)
        res = nlsolve(g!, x0)
        x = res.zero
        stress = x[1:6]
        R = x[7]
        X_1 = x[8:13]
        X_2 = x[14:19]
        seff = von_mises_stress(stress - X_1 - X_2)
        dotp = ((seff - (R0+R))/K_n)^n_n
        dp = dotp*material.dtime
        s = deviator(stress - X_1 - X_2)
        n = 1.5*s/seff
        n[4:end] .*= 2.0
        dvarepsilon_pl = dp*n
        mat.dplastic_strain[:] .= dvarepsilon_pl
        mat.dcumulative_equivalent_plastic_strain = dp
        mat.dbackstress1[:] .= X_1 - mat.backstress1
        mat.dbackstress2[:] .= X_2 - mat.backstress2
        mat.dR = R-mat.R
        dstress[:] .= stress - material.stress
        D[:,:] .= D - (D*n*n'*D) / (n'*D*n)
    end
    return nothing
end

material_preprocess_analysis!(material::Material{Chaboche}, element::Element{Poi1}, ip, time) = nothing
material_postprocess_analysis!(material::Material{Chaboche}, element::Element{Poi1}, ip, time) = nothing
material_preprocess_increment!(material::Material{Chaboche}, element::Element{Poi1}, ip, time) = nothing
material_postprocess_increment!(material::Material{Chaboche}, element::Element{Poi1}, ip, time) = nothing
material_preprocess_iteration!(material::Material{Chaboche}, element::Element{Poi1}, ip, time) = nothing
material_postprocess_iteration!(material::Material{Chaboche}, element::Element{Poi1}, ip, time) = nothing


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

function create_nonlinear_system_of_equations(material_::Material{Chaboche}, dvarepsilon_tot::Vector{Float64}, dt::Float64)
    material = material_.properties
    D = material_.jacobian # Should be updated
    K_n = material.K_n
    n_n = material.n_n
    C_1 = material.C_1
    D_1 = material.D_1
    C_2 = material.C_2
    D_2 = material.D_2
    Q = material.Q
    b = material.b
    R0 = material.yield_stress
    X_1n = material.backstress1
    X_2n = material.backstress2
    sigma_n = material_.stress
    R_n = material.R
    function g!(F, x) # System of non-linear equations
        sigma = x[1:6]
        R = x[7]
        X_1 = x[8:13]
        X_2 = x[14:19]
        seff = von_mises_stress(sigma - X_1 - X_2)
        f = seff - (R0+R)
        dotp = ((f >= 0.0 ? f : 0.0)/K_n)^n_n
        dp = dotp*dt
        s = deviator(sigma - X_1 - X_2)
        n = 1.5*s/seff
        n[4:end] .*= 2.0
        dvarepsilon_pl = dp*n
        f1 = sigma_n - sigma + D*(dvarepsilon_tot - dvarepsilon_pl)
        f2 = R_n - R + b*(Q-R)*dp
        if isapprox(C_1, 0.0)
            f3 = X_1n - X_1
        else
            f3 = X_1n - X_1 + 2.0/3.0*C_1*dp*(n - 1.5*D_1/C_1*X_1)
        end
        if isapprox(C_2, 0.0)
            f4 = X_2n - X_2
        else
            f4 = X_2n - X_2 + 2.0/3.0*C_2*dp*(n - 1.5*D_2/C_2*X_2)
        end
        F[:] = [f1; f2; f3; f4]
    end
    return g!
end
