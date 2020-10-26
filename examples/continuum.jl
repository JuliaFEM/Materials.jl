# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE
#
# Low-level definitions for one_elem_disp_chaboche.jl.

mutable struct Continuum3D <: FieldProblem
    material_model :: Symbol
end

Continuum3D() = Continuum3D(:PerfectPlastic)
FEMBase.get_unknown_field_name(::Continuum3D) = "displacement"

function FEMBase.assemble_elements!(problem::Problem{Continuum3D},
                                    assembly::Assembly,
                                    elements::Vector{Element{Hex8}},
                                    time::Float64)


    for element in elements
        for ip in get_integration_points(element)
            material = ip("material", time)
            preprocess_increment!(material, element, ip, time)
        end
    end
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

    dtime = 0.05
    # super dirty hack
    # data = first(elements).fields["displacement"].data
    # if length(data) > 1
    #     time0 = data[end-1].first
    #     dtime = time - time0
    # end

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
            integrate_material!(material)
            D = material.jacobian
            S = material.stress + material.dstress
            #@info("material matrix", D)

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

function FEMBase.assemble_elements!(problem::Problem{Continuum3D},
                                    assembly::Assembly,
                                    elements::Vector{Element{Quad4}},
                                    time::Float64)

    nnodes = 4
    ndofs = 3
    f = zeros(nnodes*ndofs)
    bi = BasisInfo(Quad4)

    for element in elements

        fill!(f, 0.0)

        for ip in get_integration_points(element)

            J, detJ, N, dN = element_info!(bi, element, ip, time)
            w = ip.weight*detJ

            if haskey(element, "surface pressure")
                J = element(ip, time, Val{:Jacobian})'
                n = cross(J[:,1], J[:,2])
                n /= norm(n)
                # sign convention, positive pressure is towards surface
                p = element("surface pressure", ip, time)
                f += w*p*vec(n*N)
            end
        end

        gdofs = get_gdofs(problem, element)
        add!(assembly.f, gdofs, f)

    end

    return nothing

end
