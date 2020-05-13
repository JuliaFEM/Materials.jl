# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

@with_kw mutable struct MemoryDriverState <: AbstractMaterialState
    time :: Float64 = zero(Float64)
    strain :: SymmetricTensor{2,3} = zero(SymmetricTensor{2,3,Float64})
end

@with_kw struct MemoryParameterState <: AbstractMaterialState
    E :: Float64 = 0.0
    nu :: Float64 = 0.0
    R0 :: Float64 = 0.0
    Kn :: Float64 = 0.0
    nn :: Float64 = 0.0
    C1 :: Float64 = 0.0
    D1 :: Float64 = 0.0
    C2 :: Float64 = 0.0
    D2 :: Float64 = 0.0
    Q0 :: Float64 = 0.0
    QM :: Float64 = 0.0
    mu :: Float64 = 0.0
    b :: Float64 = 0.0
    eta :: Float64 = 0.0
    m :: Float64 = 0.0
    pt :: Float64 = 0.0
    xi :: Float64 = 0.0
end

@with_kw struct MemoryVariableState <: AbstractMaterialState
    stress :: SymmetricTensor{2,3} = zero(SymmetricTensor{2,3,Float64})
    X1 :: SymmetricTensor{2,3} = zero(SymmetricTensor{2,3,Float64})
    X2 :: SymmetricTensor{2,3} = zero(SymmetricTensor{2,3,Float64})
    plastic_strain :: SymmetricTensor{2,3} = zero(SymmetricTensor{2,3,Float64})
    cumeq :: Float64 = zero(Float64)
    R :: Float64 = zero(Float64)
    q :: Float64 = zero(Float64)
    zeta :: SymmetricTensor{2,3} = zero(SymmetricTensor{2,3,Float64})
    jacobian :: SymmetricTensor{4,3} = zero(SymmetricTensor{4,3,Float64})
end

@with_kw mutable struct Memory <: AbstractMaterial
    drivers :: MemoryDriverState = MemoryDriverState()
    ddrivers :: MemoryDriverState = MemoryDriverState()
    variables :: MemoryVariableState = MemoryVariableState()
    variables_new :: MemoryVariableState = MemoryVariableState()
    parameters :: MemoryParameterState = MemoryParameterState()
    dparameters :: MemoryParameterState = MemoryParameterState()
end

function integrate_material!(material::Memory)
    p = material.parameters
    v = material.variables
    dd = material.ddrivers
    d = material.drivers
    @unpack E, nu, R0, Kn, nn, C1, D1, C2, D2, Q0, QM, mu, b, eta, m, pt, xi = p
    mu_ = E/(2.0*(1.0+nu))
    lambda = E*nu/((1.0+nu)*(1.0-2.0*nu))

    @unpack strain, time = d
    dstrain = dd.strain
    dtime = dd.time
    @unpack stress, X1, X2, plastic_strain, cumeq, R, q, zeta, jacobian = v


    # Elastic trial
    jacobian = isotropic_elasticity_tensor(lambda, mu_)
    stress += dcontract(jacobian, dstrain)
    seff = stress - X1 - X2
    seff_dev = dev(seff)
    f = sqrt(1.5)*norm(seff_dev) - (R0 + R)
    if f > 0.0
        g! = create_nonlinear_system_of_equations(material)
        x0 = [tovoigt(stress); R; tovoigt(X1); tovoigt(X2)]
        F = similar(x0)

        res = nlsolve(g!, x0; autodiff = :forward) # Explicit update to memory-surface
        res.f_converged || error("Nonlinear system of equations with explicit surface did not converge!")
        x = res.zero
        stress = fromvoigt(SymmetricTensor{2,3,Float64}, @view x[1:6])
        R = x[7]
        X1 = fromvoigt(SymmetricTensor{2,3,Float64}, @view x[8:13])
        X2 = fromvoigt(SymmetricTensor{2,3,Float64}, @view x[14:19])

        seff = stress - X1 - X2
        seff_dev = dev(seff)
        f = sqrt(1.5)*norm(seff_dev) - (R0 + R)
        dotp = ((f >= 0.0 ? f : 0.0)/Kn)^nn
        dp = dotp*dtime
        n = sqrt(1.5)*seff_dev/norm(seff_dev)

        # Update plastic strain
        plastic_strain += dp*n
        cumeq += dp

        # Strain memory - explicit update
        JF = sqrt(1.5)*norm(dev(plastic_strain - zeta))
        FF = 2.0/3.0*JF - q
        if FF > 0.0
            nF = 1.5*dev(plastic_strain - zeta)/JF
            nnF = dcontract(n, nF)
            if nnF>0
                q += 2.0/3.0*eta*nnF*dp
                zeta += 2.0/3.0*(1.0 - eta)*nnF*nF*dp
            end
        else
            # Memory evanescence term
            if cumeq>=pt
                q += -xi*q^m*dp
            end
        end

        # Compute Jacobian
        function residuals(x)
            F = similar(x)
            g!(F, x)
            return F
        end
        drdx = ForwardDiff.jacobian(residuals, x)
        drde = zeros((length(x),6))
        drde[1:6, 1:6] = -tovoigt(jacobian)
        jacobian = fromvoigt(SymmetricTensor{4,3}, (drdx\drde)[1:6, 1:6])
    end
    variables_new = MemoryVariableState(stress = stress,
                                          X1 = X1,
                                          X2 = X2,
                                          R = R,
                                          plastic_strain = plastic_strain,
                                          cumeq = cumeq,
                                          q = q,
                                          zeta = zeta,
                                          jacobian = jacobian)
    material.variables_new = variables_new
end

function create_nonlinear_system_of_equations(material::Memory)
    p = material.parameters
    v = material.variables
    dd = material.ddrivers
    d = material.drivers
    @unpack E, nu, R0, Kn, nn, C1, D1, C2, D2, Q0, QM, mu, b, eta, m, pt, xi = p
    mu_ = E/(2.0*(1.0+nu))
    lambda = E*nu/((1.0+nu)*(1.0-2.0*nu))

    @unpack strain, time = d
    dstrain = dd.strain
    dtime = dd.time
    @unpack stress, X1, X2, plastic_strain, cumeq, R, q, zeta, jacobian = v

    function g!(F, x::Vector{T}) where {T} # Explicit update of memory surface
        jacobian = isotropic_elasticity_tensor(lambda, mu_)
        stress_ = fromvoigt(SymmetricTensor{2,3,T}, @view x[1:6])
        R_ = x[7]
        X1_ = fromvoigt(SymmetricTensor{2,3,T}, @view x[8:13])
        X2_ = fromvoigt(SymmetricTensor{2,3,T}, @view x[14:19])

        seff = stress_ - X1_ - X2_
        seff_dev = dev(seff)
        f = sqrt(1.5)*norm(seff_dev) - (R0 + R_)

        dotp = ((f >= 0.0 ? f : 0.0)/Kn)^nn
        dp = dotp*dtime
        n = sqrt(1.5)*seff_dev/norm(seff_dev)
        dstrain_plastic = dp*n
        plastic_strain_ = plastic_strain + dstrain_plastic
        # Strain memory - explicit update
        JF = sqrt(1.5)*norm(dev(plastic_strain_ - zeta))
        FF = 2.0/3.0*JF - q
        if FF > 0.0
            nF = 1.5*dev(plastic_strain_ - zeta)/JF
            nnF = dcontract(n, nF)
            if nnF>0
                q_ = q + 2.0/3.0*eta*nnF*dp
                zeta_ = zeta + 2.0/3.0*(1.0 - eta)*nnF*nF*dp
            else
                q_ = q
                zeta_ = zeta
            end
        else
            # Memory evanescence term
            p_ = cumeq + dp
            if p_>pt
                q_ = q - xi*q^m*dp
            else
                q_ = q
            end
            zeta_ = zeta
        end

        tovoigt!(view(F, 1:6), stress - stress_ + dcontract(jacobian, dstrain - dstrain_plastic))
        F[7] = R - R_ + b*((QM + (Q0 - QM)*exp(-2.0*mu*q_))-R_)*dp
        if isapprox(C1, 0.0)
            tovoigt!(view(F, 8:13), X1 - X1_)
        else
            tovoigt!(view(F, 8:13), X1 - X1_ + 2.0/3.0*C1*dp*(n - 1.5*D1/C1*X1_))
        end
        if isapprox(C2, 0.0)
            tovoigt!(view(F, 14:19), X2 - X2_)
        else
            tovoigt!(view(F, 14:19), X2 - X2_ + 2.0/3.0*C2*dp*(n - 1.5*D2/C2*X2_))
        end
    end
    return g!
end
