# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md
using Tensors

kron_delta(i, j) = i == j ? 1 : 0
ident_func(i,j,k,l) = 1/2 * (kron_delta(i,k)*kron_delta(j,l) + kron_delta(i,l)*kron_delta(j,k))

"""
    generate_elastic_tensor_3D(E, nu)

Calculate 4th order elastic moduli
ref: https://en.wikipedia.org/wiki/Hooke's_law
"""
function generate_elastic_tensor(E, nu)
    # Lame constants
    lambda = (E * nu) / ((1 + nu) * (1 - 2*nu))
    my = E / (2*(1 + nu))

    # Identity tensor
    ident_tensor = one(Tensor{2, 3})
    II = Tensor{4, 3, Float64}(ident_func)

    # Elastic moduli
    lambda * otimes(ident_tensor, ident_tensor) + 2*my*II
end

"""
    generate_elastic_tensor_plane_stress(E, nu) 

Calculate 4th order elastic moduli for plane stress
formulation. 
ref: https://ocw.mit.edu/courses/mechanical-engineering/2-080j-structural-mechanics-fall-2013/course-notes/MIT2_080JF13_Lecture4.pdf
"""
function generate_elastic_tensor_plane_stress(E, nu)

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
    e3 = basevec(Vec{3}, 3)
    ident_tensor = one(Tensor{2,3}) - otimes(e3, e3)

    # Remove all sigma_xz, sigma_yz and sigma_zz elements
    g = (i,j,k,l) -> case(k, l) ? 0.0 : ident_func(i,j,k,l)
    II = Tensor{4, 3, Float64}(g)

    # Elastic moduli
    lambda * otimes(ident_tensor, ident_tensor) + 2*my*II

end

generate_elastic_tensor(mat::IsotropicHooke, ::Type{Val{:plane_stress}}) = generate_elastic_tensor_plane_stress(mat.youngs_modulus, mat.nu)
generate_elastic_tensor(mat::IsotropicHooke, ::Type{Val{:plane_strain}}) = generate_elastic_tensor(mat.youngs_modulus, mat.nu)
generate_elastic_tensor(mat::IsotropicHooke) = generate_elastic_tensor(mat.youngs_modulus, mat.nu)

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
    Tensor{2, 3}(strain_mat)
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
    calc_response(mat::Material, gradu)

Calculate corresponding stress for a given strain
"""
function calc_response(mat::Material, gradu)
    dim = mat.dimension
    elastic = mat.properties["elastic"]
    finite_strain = mat.finite_strain

    if dim == 1
        error("Not implemented yet")

    elseif dim == 2
        D = generate_elastic_tensor(elastic, Val{mat.formulation})
        strain = generate_strain(gradu, 2, finite_strain)
        F = generate_deformation_gradient(gradu, 2, finite_strain)

    elseif dim == 3
        D = generate_elastic_tensor(elastic)
        strain = generate_strain(gradu, 3, finite_strain)
        F = generate_deformation_gradient(gradu, 3, finite_strain)
        
    end
    stress = dcontract(D, strain)
end