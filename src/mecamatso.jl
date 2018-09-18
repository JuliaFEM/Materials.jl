# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

"""
    Continuum3D

A simplified 3D continuum model for material tests.
"""
mutable struct Continuum3D <: FieldProblem
    material_model :: Symbol
end

Continuum3D() = Continuum3D(:IdealPlastic)
FEMBase.get_unknown_field_name(::Continuum3D) = "displacement"

abstract type AbstractLoading end

mutable struct AxialStrainLoading <: AbstractLoading
    times :: Vector{Float64}
    loads :: Vector{Float64}
end
mutable struct ShearStrainLoading <: AbstractLoading
    times :: Vector{Float64}
    loads :: Vector{Float64}
end

function update_bc_elements!(bc_elements::Vector{Element{Poi1}}, loading::AxialStrainLoading)
    bc_element_1, bc_element_2, bc_element_3, bc_element_4, bc_element_5, bc_element_6, bc_element_7, bc_element_8 = bc_elements
    # Fix bottom side
    for element in (bc_element_1, bc_element_2, bc_element_3, bc_element_4)
        update!(element, "fixed displacement 3", 0.0)
    end
    # Set loading bc
    for element in (bc_element_5, bc_element_6, bc_element_7, bc_element_8)
        for (t,f) in zip(loading.times, loading.loads)
            # @info "Setting displacement 3 at $(t) to $(f)!"
            update!(element, "fixed displacement 3", t => f)
        end
    end
    # Set rest of boundary conditions
    update!(bc_element_1, "fixed displacement 1", 0.0)
    update!(bc_element_1, "fixed displacement 2", 0.0)
    update!(bc_element_2, "fixed displacement 2", 0.0)
    update!(bc_element_4, "fixed displacement 1", 0.0)
    update!(bc_element_5, "fixed displacement 1", 0.0)
    update!(bc_element_5, "fixed displacement 2", 0.0)
    update!(bc_element_6, "fixed displacement 2", 0.0)
    update!(bc_element_8, "fixed displacement 1", 0.0)
    return nothing
end

function update_bc_elements!(bc_elements::Vector{Element{Poi1}}, loading::ShearStrainLoading)
    bc_element_1, bc_element_2, bc_element_3, bc_element_4, bc_element_5, bc_element_6, bc_element_7, bc_element_8 = bc_elements
    # Fix bottom side
    for element in (bc_element_1, bc_element_2, bc_element_3, bc_element_4)
        update!(element, "fixed displacement 1", 0.0)
        update!(element, "fixed displacement 2", 0.0)
        update!(element, "fixed displacement 3", 0.0)
    end
    # Set top side bcs
    for element in (bc_element_5, bc_element_6, bc_element_7, bc_element_8)
        for (t,f) in zip(loading.times, loading.loads)
            update!(element, "fixed displacement 1", t => f)
        end
        update!(element, "fixed displacement 2", 0.0)
        update!(element, "fixed displacement 3", 0.0)
    end
    return nothing
end

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

            # Material stiffness matrix
            Km += w*BL'*D*BL

            # Internal force vector
            f_int += w*BL'*S

            # External force vector
            for i=1:dim
                haskey(element, "displacement load $i") || continue
                b = element("displacement load $i", ip, time)
                f_ext[i:dim:end] += w*b*vec(N)
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

function FEMBase.assemble_elements!(problem::Problem{Continuum3D},
                                    assembly::Assembly,
                                    elements::Vector{Element{Poi1}},
                                    time::Float64)

    u = zeros(3)
    ip = (0.0, 0.0, 0.0)

    for element in elements
        fill!(u, 0.0)
        gdofs = get_gdofs(problem, element)
        if haskey(element, "displacement")
            u[:] = element("displacement", ip, time)
        end
        for j in 1:3
            if haskey(element, "fixed displacement $j")
                dof = gdofs[j]
                add!(assembly.C1, dof, dof, 1.0)
                add!(assembly.C2, dof, dof, 1.0)
                du = element("fixed displacement $j", ip, time) - u[j]
                add!(assembly.g, dof, du)
            end
        end
    end

    return nothing

end

""" Mechanical material model analysis."""
mutable struct MecaMatSo <: AbstractAnalysis
    t0 :: Float64
    t1 :: Float64
    dt :: Float64
    u :: Vector{Float64}
    du :: Vector{Float64}
    la :: Vector{Float64}
    dla :: Vector{Float64}
    convergence_tolerance :: Float64
    max_iterations :: Int64
    extrapolate_initial_guess :: Bool
end

function MecaMatSo()
    t0 = 0.0
    t1 = 1.0
    dt = 0.1
    u = Vector{Float64}()
    du = Vector{Float64}()
    la = Vector{Float64}()
    dla = Vector{Float64}()
    convergence_tolerance = 1.0e-5
    max_iterations = 10
    extrapolate_initial_guess = false
    return MecaMatSo(t0, t1, dt, u, du, la, dla,
                     convergence_tolerance, max_iterations,
                     extrapolate_initial_guess)
end

function FEMBase.run!(analysis::Analysis{MecaMatSo})

    props = analysis.properties
    time = props.t0
    dtime = props.dt
    u = props.u
    du = props.du
    la = props.la
    dla = props.dla

    for problem in get_problems(analysis)
        FEMBase.initialize!(problem, time)
    end

    for problem in get_problems(analysis)
        for element in get_elements(problem)
            for ip in get_integration_points(element)
                material_type = getfield(Materials, problem.properties.material_model)
                material = Material(material_type, tuple())
                ip.fields["material"] = field(material)
                material_preprocess_analysis!(material, element, ip, time)
            end
        end
    end

    solution_norm = 0.0
    solution_norm_prev = Inf

    while time < props.t1

        if time + dtime > props.t1
            dtime = props.t1 - time
        end
        time = round(time + dtime, digits=12)

        if props.extrapolate_initial_guess
            # extrapolate next solution, use linear approximation
            if length(du) != 0
                nnodes = Int(length(u)/3)
                u_dict = Dict(j => [u[3*(j-1)+k] for k in 1:3] for j in 1:nnodes)
                du_dict = Dict(j => [du[3*(j-1)+k] for k in 1:3] for j in 1:nnodes)
                for problem in get_problems(analysis)
                    for element in get_elements(problem)
                        connectivity = get_connectivity(element)
                        ue = tuple(collect((u_dict[j]+du_dict[j]) for j in connectivity)...)
                        update!(element, "displacement", time => ue)
                    end
                end
            end
        else
            fill!(du, 0.0)
            solution_norm_prev = Inf
        end

        for problem in get_problems(analysis)
            for element in get_elements(problem)
                for ip in get_integration_points(element)
                    material = ip("material", time)
                    material_preprocess_increment!(material, element, ip, time)
                end
            end
        end

        n = 0
        while true

            n += 1
            @info("Solving for time $time, iteration # $n")

            for problem in get_problems(analysis)
                for element in get_elements(problem)
                    for ip in get_integration_points(element)
                        material = ip("material", time)
                        material_preprocess_iteration!(material, element, ip, time)
                    end
                end
            end

            K = SparseMatrixCOO()
            C = SparseMatrixCOO()
            f = SparseMatrixCOO()
            g = SparseMatrixCOO()
            for problem in get_problems(analysis)
                empty!(problem.assembly)
                assemble!(problem, time)
                append!(K, problem.assembly.K)
                append!(f, problem.assembly.f)
                append!(C, problem.assembly.C1)
                append!(g, problem.assembly.g)
            end

            dim = size(K, 1)

            if dim != length(u)
                resize!(u, dim)
                resize!(la, dim)
                resize!(du, dim)
                resize!(dla, dim)
                fill!(u, 0.0)
                fill!(la, 0.0)
                fill!(du, 0.0)
                fill!(dla, 0.0)
            end

            K = sparse(K, dim, dim)
            C = sparse(C, dim, dim)
            f = sparse(f, dim, 1)
            g = sparse(g, dim, 1)
            Kb = 1.0e36*C'*C
            fb = 1.0e36*C'*g

            du += Vector((cholesky(Symmetric(K+Kb)) \ (f+fb))[:])

            nnodes = Int(dim/3)
            u_dict = Dict(j => [u[3*(j-1)+k] for k in 1:3] for j in 1:nnodes)
            du_dict = Dict(j => [du[3*(j-1)+k] for k in 1:3] for j in 1:nnodes)
            for problem in get_problems(analysis)
                for element in get_elements(problem)
                    connectivity = get_connectivity(element)
                    ue = tuple(collect(u_dict[j]+du_dict[j] for j in connectivity)...)
                    update!(element, "displacement", time => ue)
                end
            end

            # Check convergence
            solution_norm = norm(du)
            norm_err = norm(solution_norm-solution_norm_prev)
            norm_err_rel = norm_err/(max(solution_norm, solution_norm_prev))
            if norm_err < props.convergence_tolerance
                @info("Solution converged in $n iterations.")
                break
            else
                solution_norm_prev = solution_norm
            end
            if n > props.max_iterations
                error("Model did not converge at time $time in $n iterations.")
            end

            for problem in get_problems(analysis)
                for element in get_elements(problem)
                    for ip in get_integration_points(element)
                        material = ip("material", time)
                        material_postprocess_iteration!(material, element, ip, time)
                    end
                end
            end

        end # iterations

        u += du

        for problem in get_problems(analysis)
            for element in get_elements(problem)
                for ip in get_integration_points(element)
                    material = ip("material", time)
                    material_postprocess_increment!(material, element, ip, time)
                end
            end
        end

        # JuliaFEM.write_results!(analysis, time)

        if time == props.t1
            @info("Simulation step ready.")
        end

    end # step

    for problem in get_problems(analysis)
        for element in get_elements(problem)
            for ip in get_integration_points(element)
                material = ip("material", time)
                material_postprocess_analysis!(material, element, ip, time)
            end
        end
    end

    return nothing

end # analysis

"""
    get_material_analysis(material_model)

Create a standardized material test for one element. Returns a tuple:

    (analysis, problem, element, ip1)

"""
function get_one_element_material_analysis(material_model::Symbol)

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
    update!(bc_elements, "geometry", X)

    problem = Problem(Continuum3D, "1 element problem", 3)
    problem.properties.material_model = material_model
    add_elements!(problem, elements, bc_elements)
    analysis = Analysis(MecaMatSo, "solve 1 element problem")
    add_problems!(analysis, problem)

    ip = first(get_integration_points(element))
    return analysis, problem, element, bc_elements, ip
end
