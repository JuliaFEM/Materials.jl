# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md
using Tensors

generate_elastic_tensor{P<:Elastic}(mat::P, dim) = generate_elastic_tensor(mat, dim, nothing)

"""
    generate_elastic_tensor_2D(E, nu, ::Type{Val{:plane_stress}})

Calculate elastic moduli in plain stress formulation

"""
function generate_elastic_tensor_2D(E, nu, ::Type{Val{:plane_stress}})
    D = E/(1.0 - nu^2) .* [
            1.0   nu    0.0
             nu  1.0    0.0
            0.0  0.0 1.0-nu]
    Tensor{2, 3}(D)
end

"""
    generate_elastic_tensor_2D(E, nu, ::Type{Val{:plane_strain}})

Calculate elastic moduli in plane stress formulation
"""
function generate_elastic_tensor_2D(E, nu, ::Type{Val{:plane_strain}})
    D = E/((1.0+nu)*(1.0-2.0*nu)) .* [
            1.0-nu      nu         0.0
                nu  1.0-nu         0.0
                0.0     0.0 1.0-2.0*nu]
    Tensor{2, 3}(D)
end

"""
    generate_elastic_tensor_3D(E, nu)

Calculate 4th order elastic moduli
ref: https://en.wikipedia.org/wiki/Lam%C3%A9_parameters
"""
function generate_elastic_tensor_3D(E, nu)
    
    # Lame constants
    kron_delta = (i, j) -> i == j ? 1 : 0
    
    # Lame constants
    lambda = (E * nu) / ((1 + nu) * (1 - 2*nu))
    my = E / (2*(1 + nu))

    # Identity tensor
    ident_tensor = one(Tensor{2, 3})
    f = (i,j,k,l) -> 1/2 * (kron_delta(i,k)*kron_delta(j,l) + kron_delta(i,l)*kron_delta(j,k))
    II = Tensor{4, 3, Float64}(f)

    # Elastic moduli
    lambda * otimes(ident_tensor, ident_tensor) + 2*my*II
end

"""
    generate_elastic_tensor{P<:Elastic}(mat::P, dim, formulation)

Calculate elastic moduli
"""
function generate_elastic_tensor{P<:Elastic}(mat::P, dim, formulation)
    E = mat.youngs_modulus
    nu = mat.nu
    if dim == 1
        return E
    elseif dim == 2
        return generate_elastic_tensor_2D(E, nu, Val{formulation})
    elseif dim == 3
        return generate_elastic_tensor_3D(E, nu)
    else
        error("Did not understand dimension. Given dimension $dim.")
    end
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
    Tensor{2, dim}(strain)
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
    Tensor{2, dim}(F)
end

"""
    calculate_2D(mat::Material, gradu)

Calculate stress either in plane stress or plain strain formulation
"""
function calculate_2D(mat::Material, gradu)
    formulation = mat.formulation
    finite_strain = mat.finite_strain
    elastic_properties = mat.properties["elastic"]    
    D = generate_elastic_tensor(elastic_properties, 2, formulation)
    F = generate_deformation_gradient(gradu, 2, finite_strain)
    strain = generate_strain(gradu, 2, finite_strain)
    strain_vec = Tensor{1, 3}([strain[1,1], strain[2,2], strain[1,2]])
    dot(D, strain_vec)
end

"""
    calculate_3D(mat::Material, gradu)

Calculate stress either in plane stress or plain strain formulation
"""
function calculate_3D(mat::Material, gradu)
    finite_strain = mat.finite_strain
    elastic_properties = mat.properties["elastic"]
    D = generate_elastic_tensor(elastic_properties, 3)
    F = generate_deformation_gradient(gradu, 3, finite_strain)
    strain = generate_strain(gradu, 3, finite_strain)
    dcontract(D, strain)
end

"""
    calc_response(mat::Material, gradu)

Calculate corresponding stress for a given strain
"""
function calc_response(mat::Material, gradu)
    dim = mat.dimension
    if dim == 1
        error("Not implemented yet")
    elseif dim == 2
        stress = calculate_2D(mat, gradu)
    elseif dim == 3
        stress = calculate_3D(mat, gradu)
    end
    stress
end