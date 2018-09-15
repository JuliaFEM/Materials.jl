# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

kron_delta(i::Integer, j::Integer) = i == j ? 1 : 0
ident_func(i,j,k,l) = 1/2 * (kron_delta(i,k)*kron_delta(j,l) + kron_delta(i,l)*kron_delta(j,k))

function eye(m::Int64)
    Matrix(1.0I, m, m)
end

function init_identity_fourth_order()
    retval = Array{Float64, 4}(undef, 3, 3, 3, 3)
    @einsum retval[i,j,k,l] = ident_func(i,j,k,l)
    return retval
end

function otimes(x, y)
    retval = Array{Float64, 4}(undef, 3, 3, 3, 3)
    @einsum retval[i,j,k,l] = x[i,j]*y[k,l]
    return retval
end

"""
    calc_elastic_tensor_3D(E, nu)

Calculate 4th order elastic moduli
ref: https://en.wikipedia.org/wiki/Hooke's_law
"""
function calc_elastic_tensor(E, nu)
    # Lame constants
    lambda = (E * nu) / ((1 + nu) * (1 - 2*nu))
    my = E / (2*(1 + nu))

    # Identity tensor
    ident_tensor = Matrix(1.0I,3,3)
    II = init_identity_fourth_order()

    ident_fourth = otimes(ident_tensor, ident_tensor)

    # Elastic moduli
    lambda * ident_fourth + 2 * my * II
end

"""
    calc_elastic_tensor_plane_stress(E, nu)

Calculate 4th order elastic moduli for plane stress
formulation.
ref: https://ocw.mit.edu/courses/mechanical-engineering/2-080j-structural-
mechanics-fall-2013/course-notes/MIT2_080JF13_Lecture4.pdf
"""
function calc_elastic_tensor_plane_stress(E, nu)

    """
        Remove all the sigma_xz, sigma_yz and sigma_zz elements
    """
    function case(i, k)
        (i == 3 && k == 1 ||
         i == 3 && k == 2 ||
         i == 1 && k == 3 ||
         i == 2 && k == 3 ||
         i == 3 && k == 3)
    end

    # Lame constant
    my = E / (2*(1 + nu))

    # Compensation for plane stress
    lambda = (E * nu) / (1 - nu^2)

    # Identity tensor, but do not include sigma_zz element
    ident_tensor = Matrix(1.0I,3,3)
    ident_tensor[3, 3] -= 1

    # Remove all sigma_xz, sigma_yz and sigma_zz elements
    g = (i,j,k,l) -> case(k, l) ? 0.0 : ident_func(i,j,k,l)
    II = Array{Float64, 4}(undef, 3, 3, 3, 3)
    @einsum II[i,j,k,l] = g(i,j,k,l)

    ident_fourth = otimes(ident_tensor, ident_tensor)

    # Elastic moduli
    lambda * ident_fourth + 2 * my * II
end

function calc_elastic_tensor(mat::IsotropicHooke, ::Type{Val{:plane_stress}})
    calc_elastic_tensor_plane_stress(mat.youngs_modulus, mat.nu)
end

function calc_elastic_tensor(mat::IsotropicHooke, ::Type{Val{:plane_strain}})
    calc_elastic_tensor(mat.youngs_modulus, mat.nu)
end

function calc_elastic_tensor(mat::IsotropicHooke, ::Type{Val{:basic}})
    calc_elastic_tensor(mat.youngs_modulus, mat.nu)
end

"""
    generate_strain(gradu, dim, finite_strain::Bool)

Calculate string using gradu
"""
function generate_strain(gradu, dim, finite_strain::Bool)
    if finite_strain
        strain = 1/2*(gradu + gradu' + gradu'*gradu)
    else
       strain = 1/2*(gradu + gradu')
    end
    if dim == 2
        strain_mat = zeros(3,3)
        strain_mat[1:2, 1:2] = strain[:,:]
    else
        strain_mat = strain
    end
    strain_mat
end

"""
    generate_deformation_gradient(gradu, dim, finite_strain::Bool)

Calculate deformation gradient
"""
function generate_deformation_gradient(gradu, dim, finite_strain::Bool)
    if finite_strain
        F = eye(dim) + gradu
    else
        F = eye(dim)
    end
    F
end

function plastic_response(plastic_model::VonMises, stress_trial, stress_last, dstrain, D)
    if yield_function(stress_trial, plastic_model) > 0

        yield(stress) = yield_function(stress, plastic_model)
        dyield(stress) = d_yield_function(stress, plastic_model)
        params = Dict{AbstractString, Any}()
        params["yield_function"] = yield
        params["init_stress"] = stress_last
        params["dstrain"] = dstrain
        params["d_yield_function"] = dyield
        params["D"] = D
        x = rand(7)
        x[end] = 0.0

        f(stress_) = radial_return(stress_, params)
        df(stress_) = ForwardDiff.jacobian(f, stress_)
        root = find_root(f, df, x; max_iter=50, norm_acc=1e-9)

        dstress = array_to_tensor(root)
        stress_trial = stress_last + dstress
    end
    stress_trial
end

"""Calculate stress response for given material model
"""
function calculate_stress!(mat::Material, model, gradu, dim, finite_strain, formulation)
    D = calc_elastic_tensor(model.elastic, Val{formulation})
    dstrain = generate_strain(gradu, dim, finite_strain)
    dstress = dcontract(D, dstrain, Val{:DiffSize})

    stress_last = mat.history_values["stress"][end]
    strain_last = mat.history_values["strain"][end]

    stress_trial = stress_last + dstress
    strain = strain_last + dstrain
    F = generate_deformation_gradient(gradu, dim, finite_strain)

    plastic_model = model.plastic
    if typeof(plastic_model) == NoPlasticity
        stress = stress_trial
    else
        stress = plastic_response(plastic_model, stress_trial, stress_last, dstrain, D)
    end
    mat.trial_values["stress"] = stress
    mat.trial_values["strain"] = strain
    mat.trial_values["F"] = F
end

"""
    calc_response(mat::Material, gradu)

Calculate corresponding stress for a given strain
"""
function calc_response!(mat::Material, gradu)
    dim = mat.dimension
    model = mat.model
    finite_strain = mat.finite_strain
    formulation = mat.formulation
    calculate_stress!(mat, model, gradu, dim, finite_strain, formulation)
end
