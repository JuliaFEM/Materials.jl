# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using FEMBase, LinearAlgebra

mutable struct Continuum3D <: FieldProblem
    material_model :: Symbol
end

Continuum3D() = Continuum3D(:IdealPlastic)
FEMBase.get_unknown_field_name(::Continuum3D) = "displacement"

function FEMBase.assemble_elements!(problem::Problem{Continuum3D},
                                    assembly::Assembly,
                                    elements::Vector{Element{Hex8}},
                                    time::Float64)

    bi = BasisInfo(Hex8)

    dim = 3
    nnodes = 8
    ndofs = dim*nnodes

    BL = zeros(6, ndofs)
    Km = zeros(ndofs, ndofs)
    f_int = zeros(ndofs)
    f_ext = zeros(ndofs)
    D = zeros(6, 6)
    S = zeros(6)

    # dirty hack
    time0 = first(elements).fields["displacement"].data[end-1].first
    dtime = time - time0

    for element in elements

        u = element("displacement", time)
        fill!(Km, 0.0)
        fill!(f_int, 0.0)
        fill!(f_ext, 0.0)

        for ip in get_integration_points(element)
            J, detJ, N, dN = element_info!(bi, element, ip, time)
            material = ip("material", time)
            w = ip.weight*detJ

            # Kinematic matrix, linear part

            fill!(BL, 0.0)
            for i=1:nnodes
                BL[1, 3*(i-1)+1] = dN[1,i]
                BL[2, 3*(i-1)+2] = dN[2,i]
                BL[3, 3*(i-1)+3] = dN[3,i]
                BL[4, 3*(i-1)+1] = dN[2,i]
                BL[4, 3*(i-1)+2] = dN[1,i]
                BL[5, 3*(i-1)+2] = dN[3,i]
                BL[5, 3*(i-1)+3] = dN[2,i]
                BL[6, 3*(i-1)+1] = dN[3,i]
                BL[6, 3*(i-1)+3] = dN[1,i]
            end

            # Calculate stress response
            calculate_stress!(material, element, ip, time, dtime, D, S)
            update!(ip, "stress", time => S)

            # Material stiffness matrix
            Km += w*BL'*D*BL

            # Internal force vector
            f_int += w*BL'*S

            # External force vector
            for i=1:dim
                haskey(element, "displacement load $i") || continue
                b = element("displacement load $i", ip, time)
                f_ext[i:dim:end] += w*B*vec(N)
            end

        end

        # add contributions to K, Kg, f
        gdofs = get_gdofs(problem, element)
        add!(assembly.K, gdofs, gdofs, Km)
        add!(assembly.f, gdofs, f_ext - f_int)

    end

    return nothing
end

abstract type AbstractMaterial end

mutable struct IdealPlastic <: AbstractMaterial
    # Material parameters
    youngs_modulus :: Float64
    poissons_ratio :: Float64
    yield_stress :: Float64
    # Internal state variables
    plastic_strain :: Matrix{Float64}
    plastic_multiplier :: Float64
end

function IdealPlastic(element, ip, time)
    # Material parameters
    youngs_modulus = element("youngs modulus", ip, time)
    poissons_ratio = element("poissons ratio", ip, time)
    yield_stress = element("yield stress", ip, time)
    # Internal variables
    plastic_strain = element("plastic strain", ip, time)
    plastic_multiplier = 0.0
    return IdealPlastic(youngs_modulus, poissons_ratio, yield_stress,
                        plastic_strain, plastic_multiplier)
end

function calculate_stress!(material::AbstractMaterial, element, ip, time, dtime,
                           material_matrix, stress_vector)
    # Update material parameters
    material.youngs_modulus = element("youngs modulus", ip, time)
    material.poissons_ratio = element("poissons ratio", ip, time)
    material.yield_stress = element("yield stress", ip, time)

    gradu0 = element("displacement", ip, time-dtime, Val{:Grad})
    gradu = element("displacement", ip, time, Val{:Grad})

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
    @info("stuff", stress_v, material.yield_stress)

    if stress_v < material.yield_stress
        @info("Elastic strain")
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
        stress_vector[1] = stress_trial[1,1]
        stress_vector[2] = stress_trial[2,2]
        stress_vector[3] = stress_trial[3,3]
        stress_vector[4] = stress_trial[1,2]
        stress_vector[5] = stress_trial[2,3]
        stress_vector[6] = stress_trial[3,1]
    else
        @info("Plastic strain")
        n = 3.0/2.0*stress_dev/stress_v
        dla = (stress_v - material.yield_stress)/(3.0*G)
        dstrain_pl = dla*n
        material.plastic_strain += dstrain_pl
        material.plastic_multiplier += dla
        fill!(material_matrix, 0.0)
        dstrain_el = dstrain - dstrain_pl
        dstress = lambda*tr(dstrain_el)*I + 2.0*mu*dstrain_el
        stress = stress0 + dstress
        stress_vector[1] = stress[1,1]
        stress_vector[2] = stress[2,2]
        stress_vector[3] = stress[3,3]
        stress_vector[4] = stress[1,2]
        stress_vector[5] = stress[2,3]
        stress_vector[6] = stress[3,1]
    end

    return nothing
end


X = Dict(
    1 => [0.0, 0.0, 0.0],
    2 => [1.0, 0.0, 0.0],
    3 => [1.0, 1.0, 0.0],
    4 => [0.0, 1.0, 0.0],
    5 => [0.0, 0.0, 1.0],
    6 => [1.0, 0.0, 1.0],
    7 => [1.0, 1.0, 1.0],
    8 => [0.0, 1.0, 1.0])

u0 = Dict(j => zeros(3) for j in 1:8)
u1 = copy(u0)
u1[7] = [1.0, 0.0, 0.0]

element = Element(Hex8, (1, 2, 3, 4, 5, 6, 7, 8))
update!(element, "geometry", X)
update!(element, "displacement", 0.0 => u0)
update!(element, "displacement", 1.0 => u1)
update!(element, "youngs modulus", 100.0)
update!(element, "poissons ratio", 0.3)
update!(element, "yield stress", 50.0)
update!(element, "plastic strain", 0.0 => zeros(3,3))

# Initialize material model to integration points
for ip in get_integration_points(element)
    ip.fields["material"] = field(IdealPlastic(element, ip, 0.0))
    update!(ip, "stress", 0.0 => zeros(6))
end

function Base.run()
    time = 0.0
    time_end = 1.0
    dtime = 0.1
    material_matrix = zeros(6,6)
    stress_vector = zeros(6)
    while time < time_end
        time += dtime
        @info("time = $time")
        for ip in get_integration_points(element)
            material = ip("material", time)
            calculate_stress!(material, element, ip, time, dtime,
                              material_matrix, stress_vector)
            update!(ip, "stress", time => stress_vector)
        end
    end
end

run()

using Plots
ip1 = first(get_integration_points(element))
t = range(0, stop=1.0, length=50)
s11(t) = ip1("stress", t)[1]
s22(t) = ip1("stress", t)[2]
s33(t) = ip1("stress", t)[3]
s12(t) = ip1("stress", t)[4]
s23(t) = ip1("stress", t)[5]
s31(t) = ip1("stress", t)[6]
s(t) = ip1("stress", t)
labels = ["s11" "s22" "s33" "s12" "s23" "s31"]
plot(t, s11, title="stress at integration point 1", label="s11")
plot!(t, s22, label="s22")
plot!(t, s33, label="s33")
plot!(t, s12, label="s12")
plot!(t, s23, label="s23")
plot!(t, s31, label="s31")
