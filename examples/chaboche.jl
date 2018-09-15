# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE
using Einsum, LinearAlgebra, NLsolve

I_ = Matrix(1.0I,3,3) # Second order identity tensor
II = zeros(3,3,3,3)
@einsum II[i,j,k,l] = 0.5*(I_[i,k]*I_[j,l] + I_[i,l]*I_[j,k]) # Fourth order symmetric identity tensor
IxI = zeros(3,3,3,3)
@einsum IxI[i,j,k,l] = I_[i,j]*I_[k,l] # "Trace" tensor
P = II - 1/3*IxI # Deviatoric projection tensor
function double_contraction(x::AbstractArray{<:Number,2},y::AbstractArray{<:Number,2})
    return sum(x.*y)
end

function double_contraction(x::AbstractArray{<:Number,4},y::AbstractArray{<:Number,2})
    retval = zeros(3,3)
    @einsum retval[i,j] = x[i,j,k,l]*y[k,l]
    return retval
end

function deviator(x::AbstractArray{<:Number,2})
    s = zeros(3,3)
    @einsum s[i,j] = P[i,j,k,l]*x[k,l]
    return s
end

function von_mises_stress(stress::AbstractArray{<:Number,2})
    s = deviator(stress)
    return sqrt(3/2*double_contraction(s,s))
end

function fromvector(x; offdiag=1.0)
    return [x[1] x[4]/offdiag x[6]/offdiag;
           x[4]/offdiag x[2] x[5]/offdiag;
           x[6]/offdiag x[5]/offdiag x[3]]
end

function fromtensor(x; offdiag=1.0)
    return [x[1,1], x[2,2], x[3,3], offdiag*x[1,2], offdiag*x[2,3], offdiag*x[1,3]]
end

function tensor_to_matrix(A)
    B = zeros(6,6)
    for i in 1:6
        if i<=3
            j = k = i
        else
            if i==4
                j = 1
                k = 2
            elseif i==5
                j = 2
                k = 3
            else
                j = 1
                k = 3
            end
        end
        B[i,1] = A[j,k,1,1]
        B[i,2] = A[j,k,2,2]
        B[i,3] = A[j,k,3,3]
        B[i,4] = A[j,k,1,2]
        B[i,5] = A[j,k,2,3]
        B[i,6] = A[j,k,1,3]
    end
    return B
end

abstract type AbstractMaterial end

mutable struct Chaboche <: AbstractMaterial
    # Material parameters
    youngs_modulus :: Float64
    poissons_ratio :: Float64
    K_n :: Float64
    n_n :: Float64
    C_1 :: Float64
    D_1 :: Float64
    C_2 :: Float64
    D_2 :: Float64
    Q :: Float64
    b :: Float64
    # Internal state variables
    stress :: Array{Float64,1}
    plastic_strain :: Array{Float64,1}
    cumulative_equivalent_plastic_strain :: Float64
    backstress1 :: Array{Float64,1}
    backstress2 :: Array{Float64,1}
    yield_stress :: Float64
end

function Chaboche(element, ip, time)
    # Material parameters
    youngs_modulus = element("youngs modulus", ip, time)
    poissons_ratio = element("poissons ratio", ip, time)
    K_n = element("K_n", ip, time)
    n_n = element("n_n", ip, time)
    C_1 = element("C_1", ip, time)
    D_1 = element("D_1", ip, time)
    C_2 = element("C_2", ip, time)
    D_2 = element("D_2", ip, time)
    Q = element("Q", ip, time)
    b = element("b", ip, time)

    # Internal variables
    # stress = element("stress", ip, time)
    # plastic_strain = element("plastic strain", ip, time))
    # cumulative_equivalent_plastic_strain = element("cumulative equivalent plastic strain", ip, time)
    # backstress1 = element("backstress 1", ip, time))
    # backstress2 = element("backstress 2", ip, time))
    # yield_stress = element("yield stress", ip, time)
    stress = ip("stress", time)
    plastic_strain = ip("plastic strain", time)
    cumulative_equivalent_plastic_strain = ip("cumulative equivalent plastic strain", time)
    backstress1 = ip("backstress 1", time)
    backstress2 = ip("backstress 2", time)
    yield_stress = ip("yield stress", time)
    @info "cumeq($time) = $cumulative_equivalent_plastic_strain"
    return Chaboche(youngs_modulus, poissons_ratio, K_n, n_n, C_1, D_1, C_2, D_2,
                    Q, b, stress, plastic_strain, cumulative_equivalent_plastic_strain,
                    backstress1, backstress2, yield_stress)
end

function create_nonlinear_system_of_equations(material::Chaboche, dvarepsilon_tot::AbstractArray{<:Number,2}, dt::Float64)
    global I_, P, IxI, II
    E = material.youngs_modulus
    nu = material.poissons_ratio
    mu = E/(2.0*(1.0+nu))
    lambda = E*nu/((1.0+nu)*(1.0-2.0*nu))
    K = E/(3*(1-2*nu))
    G = 0.5*E/(1.0+nu)
    C = K*IxI + 2*G*P
    K_n = material.K_n
    n_n = material.n_n
    C_1 = material.C_1
    D_1 = material.D_1
    C_2 = material.C_2
    D_2 = material.D_2
    Q = material.Q
    b = material.b
    R_n = material.yield_stress
    X_1n = fromvector(material.backstress1)
    X_1n = fromvector(material.backstress1)
    X_2n = fromvector(material.backstress2)
    sigma_n = fromvector(material.stress)
    function g!(F, x) # System of non-linear equations
        sigma = reshape(x[1:9], 3,3)
        R = x[10]
        X_1 = reshape(x[11:19], 3,3)
        X_2 = reshape(x[20:28], 3,3)
        dotp = ((von_mises_stress(sigma - X_1 - X_2) - R)/K_n)^n_n
        dp = dotp*dt
        s = deviator(sigma - X_1 - X_2)
        n = 3/2*s/von_mises_stress(sigma - X_1 - X_2)
        dvarepsilon_pl = dp*n
        material.yield_stress = R
        material.backstress1 = fromtensor(X_1)
        material.backstress2 = fromtensor(X_2)
        material.stress = fromtensor(sigma)
        material.plastic_strain .+= fromtensor(dvarepsilon_pl; offdiag=2.0)
        material.cumulative_equivalent_plastic_strain += dp
        f1 = vec(sigma_n - sigma + double_contraction(C, dvarepsilon_tot - dvarepsilon_pl))
        f2 = R_n - R + b*(Q-R)*dp
        f3 = vec(X_1n - X_1 + 2/3*C_1*dp*(n - 3*D_1/(2*C_1)*X_1))
        f4 = vec(X_2n - X_2 + 2/3*C_2*dp*(n - 3*D_2/(2*C_2)*X_2))
        F[:] = vec([f1; f2; f3; f4])
    end
    return g!
end

function calculate_stress!(material::Chaboche, element, ip, time, dtime,
                           material_matrix, stress_vector)
    # Update material parameters
    global I_, P, IxI, II
    material.youngs_modulus = element("youngs modulus", ip, time)
    material.poissons_ratio = element("poissons ratio", ip, time)
    material.K_n = element("K_n", ip, time)
    material.n_n = element("n_n", ip, time)
    material.C_1 = element("C_1", ip, time)
    material.D_1 = element("D_1", ip, time)
    material.C_2 = element("C_2", ip, time)
    material.D_2 = element("D_2", ip, time)
    material.Q = element("Q", ip, time)
    material.b = element("b", ip, time)


    material.stress = ip("stress", time - dtime)
    material.plastic_strain = ip("plastic strain", time - dtime)
    material.cumulative_equivalent_plastic_strain = ip("cumulative equivalent plastic strain", time - dtime)
    material.backstress1 = ip("backstress 1", time - dtime)
    material.backstress2 = ip("backstress 2", time - dtime)
    material.yield_stress = ip("yield stress", time - dtime)

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
    K = E/(3*(1-2*nu))
    G = 0.5*E/(1.0+nu)
    K_n = material.K_n
    n_n = material.n_n
    C_1 = material.C_1
    D_1 = material.D_1
    C_2 = material.C_2
    D_2 = material.D_2
    Q = material.Q
    b = material.b

    C = K*IxI + 2*G*P

    stress0 = fromvector(material.stress)
    # peeq = material.cumulative_equivalent_plastic_strain
    X_1 = fromvector(material.backstress1)
    X_2 = fromvector(material.backstress2)
    R = material.yield_stress

    stress_tr = stress0 + double_contraction(C, dstrain)

    f_tr = von_mises_stress(stress_tr - X_1 - X_2) - R

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
    @info "time = $time, dstrain = $dstrain, stress0 = $stress0"
    if f_tr <= 0.0
        @info "Elastic step f_tr = $f_tr !"
        stress_vector[:] .= fromtensor(stress_tr)
        material.stress = stress_vector
        return nothing
    else
        @info "Plastic step f_tr = $f_tr !"
        g! = create_nonlinear_system_of_equations(material, dstrain, dtime)
        x0 = vec([vec(stress_tr); R; vec(X_1); vec(X_2)])
        F = similar(x0)
        res = nlsolve(g!, x0)
        x = res.zero
        stress = reshape(x[1:9],3,3)
        R = x[10]
        X_1 = reshape(x[11:19], 3,3)
        X_2 = reshape(x[20:28], 3,3)
        dotp = ((von_mises_stress(stress - X_1 - X_2) - R)/K_n)^n_n
        dp = dotp*dtime
        s = deviator(stress - X_1 - X_2)
        n = 3/2*s/von_mises_stress(stress - X_1 - X_2)
        stress_vector[:] .= fromtensor(stress)
        D = material_matrix
        dg = df = fromtensor(n; offdiag=2.0)
        material_matrix[:,:] .= D - (D*dg*df'*D) / (df'*D*dg)
        material_matrix[abs.(material_matrix) .< 1.0e-9] .= 0.0
        #@info("results", material_matrix, stress0, dstrain)
    end
    return nothing
end
