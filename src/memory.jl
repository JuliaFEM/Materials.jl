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
        """
        # Iterate the full system
        x0 = [tovoigt(stress); R; tovoigt(X1); tovoigt(X2); q; tovoigt(zeta)]
        F = similar(x0)
        res = nlsolve(g!, x0; autodiff = :forward)
        x = res.zero
        if !res.f_converged
            # Try with different initial guess
            println("Iteration with elastic trial did not converge, attempting with extrapolated initial guess")
            x0 = [tovoigt(v.stress + dcontract(v.jacobian, dstrain)); R; tovoigt(X1); tovoigt(X2); q; tovoigt(zeta)]
            res = nlsolve(g!, x0; autodiff = :forward)
            x = res.zero
        end
        if !res.f_converged
            println(res)
            println("x: ", x)
            println("x0: ", x0)
            println("dstrain: ", dstrain)
        end
        res.f_converged || error("Nonlinear system of equations with did not converge!")

        stress = fromvoigt(SymmetricTensor{2,3,Float64}, @view x[1:6])
        R = x[7]
        X1 = fromvoigt(SymmetricTensor{2,3,Float64}, @view x[8:13])
        X2 = fromvoigt(SymmetricTensor{2,3,Float64}, @view x[14:19])
        q = x[20]
        zeta = fromvoigt(SymmetricTensor{2,3,Float64}, @view x[21:26])

        seff = stress - X1 - X2
        seff_dev = dev(seff)
        f = sqrt(1.5)*norm(seff_dev) - (R0 + R)
        dotp = ((f >= 0.0 ? f : 0.0)/Kn)^nn
        dp = dotp*dtime
        n = sqrt(1.5)*seff_dev/norm(seff_dev)
        # First iterate with constant memory surface
        function g0!(F, x)
            x_ = [x; q; tovoigt(zeta)]
            F_ = similar(x_)
            g!(F_, x_)
            F[:] = F_[1:19] # Only consider equations without memory surface
        end
        """
        x0 = [tovoigt(stress); R; tovoigt(X1); tovoigt(X2)]
        F = similar(x0)

        res = nlsolve(g!, x0; autodiff = :forward) # Explicit update to memory-surface
        #res = nlsolve(g0!, x0; autodiff = :forward)
        res.f_converged || error("Nonlinear system of equations without memory surface did not converge!")
        # Candidate result, holds if memory surface <= 0 and no memory-evanescence
        #x = [res.zero; q; tovoigt(zeta)]
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

        """
        # Check memory surface
        # JF = sqrt(1.5)*norm(dev(plastic_strain + dp*n - zeta))
        # FF = 2.0/3.0*JF - q
        # if FF > 0.0 # If no memory-evanescence takes place
            # println("FF: ", FF)
            # Iterate the full system with better initial guess
            x0 = [tovoigt(stress); R; tovoigt(X1); tovoigt(X2); q; tovoigt(zeta)]
            F = similar(x0)
            res = nlsolve(g!, x0; autodiff = :forward)
            #res = nlsolve(g!, x0; autodiff = :forward, ftol=1e-4)
            x = res.zero
            if !res.f_converged
                println(res)
                println("x: ", x)
                println("x0: ", x0)
            end
            #Test not raising an error here
            res.f_converged || error("Nonlinear system of equations with memory surface did not converge!")

            stress = fromvoigt(SymmetricTensor{2,3,Float64}, @view x[1:6])
            R = x[7]
            X1 = fromvoigt(SymmetricTensor{2,3,Float64}, @view x[8:13])
            X2 = fromvoigt(SymmetricTensor{2,3,Float64}, @view x[14:19])
            q = x[20]
            zeta = fromvoigt(SymmetricTensor{2,3,Float64}, @view x[21:26])

            seff = stress - X1 - X2
            seff_dev = dev(seff)
            f = sqrt(1.5)*norm(seff_dev) - (R0 + R)
            dotp = ((f >= 0.0 ? f : 0.0)/Kn)^nn
            dp = dotp*dtime
            n = sqrt(1.5)*seff_dev/norm(seff_dev)
        # end

        # Lastly, iterate with constant memory surface
        function g1!(F, x)
            x_ = [x; q; tovoigt(zeta)]
            F_ = similar(x_)
            g!(F_, x_)
            F[:] = F_[1:19] # Only consider equations without memory surface
        end
        x0 = [tovoigt(stress); R; tovoigt(X1); tovoigt(X2)]
        F = similar(x0)
        res = nlsolve(g1!, x0; autodiff = :forward)
        res.f_converged || error("Nonlinear system of equations without memory surface did not converge!")
        x = [res.zero; q; tovoigt(zeta)]
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
        """

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

    function deriv!(F, h, x::Vector{T}) where {T}
        # h \in [0,1], -> t \in time + h*dtime
        total_strain_ = strain + h*dstrain

        jacobian = isotropic_elasticity_tensor(lambda, mu_)
        compliance = isotropic_compliance_tensor(lambda, mu_)
        stress_ = fromvoigt(SymmetricTensor{2,3,T}, @view x[1:6])
        R_ = x[7]
        X1_ = fromvoigt(SymmetricTensor{2,3,T}, @view x[8:13])
        X2_ = fromvoigt(SymmetricTensor{2,3,T}, @view x[14:19])
        q_ = x[20]
        zeta_ = fromvoigt(SymmetricTensor{2,3,T}, @view x[21:26])
        elastic_strain_ = dcontract(compliance, stress_)
        plastic_strain_ = total_strain_ - elastic_strain_

        seff = stress_ - X1_ - X2_
        seff_dev = dev(seff)
        f = sqrt(1.5)*norm(seff_dev) - (R0 + R_)
        n = sqrt(1.5)*seff_dev/norm(seff_dev)
        dotp = ((f >= 0.0 ? f : 0.0)/Kn)^nn
        dstrain_plastic = dotp*n

        tovoigt!(view(F, 1:6), dcontract(jacobian, dstrain/dtime - dstrain_plastic))
        F[7] = b*((QM + (Q0 - QM)*exp(-2.0*mu*q_)) - R_)*dotp
        if isapprox(C1, 0.0)
            F[8:13] .= 0.0
        else
            tovoigt!(view(F, 8:13), 2.0/3.0*C1*dotp*(n - 1.5*D1/C1*X1_))
        end
        if isapprox(C2, 0.0)
            F[14:19] .= 0.0
        else
            tovoigt!(view(F, 14:19), 2.0/3.0*C2*dotp*(n - 1.5*D2/C2*X2_))
        end
        # Strain memory
        JF = sqrt(1.5)*norm(dev(plastic_strain_ - zeta_))
        FF = 2.0/3.0*JF - q_
        if FF > 0.0
            nF = 1.5*dev(plastic_strain_ - zeta_)/JF
            nnF = dcontract(n, nF)
            # nn = max(nnF, zero(nnF)) # This does not work, produces ForwardDiff error
            # nn = nnF > 0.0 ? nnF : 0.0 # This does not work, produces ForwardDiff error
            # F[20] = 2.0/3.0*eta*nn*dotp
            # tovoigt!(view(F, 21:26), 2.0/3.0*(1.0 - eta)*nn*nF*dotp)
            if nnF>0 # This works with ForwardDiff
                F[20] = 2.0/3.0*eta*nnF*dotp
                tovoigt!(view(F, 21:26), 2.0/3.0*(1.0 - eta)*nnF*nF*dotp)
            else
                F[20] = 0.0
                F[21:26] .= 0.0
            end
        else
            # Memory evanescence term
            F[20] = -xi*q_^m*dotp
            F[21:26] .= 0.0
        end
    end

    theta = 1.0 # Theta-method parameter \in [0,1]
    x0 = [tovoigt(stress); R; tovoigt(X1); tovoigt(X2); q; tovoigt(zeta)]
    F0 = similar(x0)
    deriv!(F0, 0.0, x0)
    function g2!(F, x::Vector{T}) where {T}
        F1 = similar(x)
        deriv!(F1, 1.0, x)
        F[:] = x0 - x + dtime*((1.0-theta)*F0 + theta*F1)
    end

    function g!(F, x::Vector{T}) where {T} # System of non-linear equations
        jacobian = isotropic_elasticity_tensor(lambda, mu_)
        stress_ = fromvoigt(SymmetricTensor{2,3,T}, @view x[1:6])
        R_ = x[7]
        X1_ = fromvoigt(SymmetricTensor{2,3,T}, @view x[8:13])
        X2_ = fromvoigt(SymmetricTensor{2,3,T}, @view x[14:19])
        q_ = x[20]
        zeta_ = fromvoigt(SymmetricTensor{2,3,T}, @view x[21:26])

        seff = stress_ - X1_ - X2_
        seff_dev = dev(seff)
        f = sqrt(1.5)*norm(seff_dev) - (R0 + R_)

        dotp = ((f >= 0.0 ? f : 0.0)/Kn)^nn
        dp = dotp*dtime
        n = sqrt(1.5)*seff_dev/norm(seff_dev)
        dstrain_plastic = dp*n
        plastic_strain_ = plastic_strain + dstrain_plastic

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
        # Strain memory
        JF = sqrt(1.5)*norm(dev(plastic_strain_ - zeta_))
        # JF = sqrt(1.5)*norm(plastic_strain_ - zeta_)
        FF = 2.0/3.0*JF - q_
        if FF > 0.0
            nF = 1.5*dev(plastic_strain_ - zeta_)/JF
            # nF = 1.5*(plastic_strain_ - zeta_)/JF
            nnF = dcontract(n, nF)
            if nnF>0
                F[20] = q - q_ + 2.0/3.0*eta*nnF*dp
                # Test replacing the evolution equation with the memory surface
                #F[20] = FF
                tovoigt!(view(F, 21:26), zeta - zeta_ + 2.0/3.0*(1.0 - eta)*nnF*nF*dp)
            else
                F[20] = q - q_
                tovoigt!(view(F, 21:26), zeta - zeta_)
                # F[20] = 0.0
                # F[21:26] .= 0.0
            end
        else
            # Memory evanescence term
            p_ = cumeq + dp
            if p_>pt
                #F[20] = q - q_ - xi*(p_ - pt)*p_^m*dp # Chaboche
                F[20] = q - q_ - xi*q_^m*dp # Nouilhas
            else
                F[20] = q - q_
                # F[20] = 0.0
            end
            tovoigt!(view(F, 21:26), zeta - zeta_)
            # F[21:26] .= 0.0
        end
    end

    function g3!(F, x::Vector{T}) where {T} # Explicit update of memory surface
        jacobian = isotropic_elasticity_tensor(lambda, mu_)
        stress_ = fromvoigt(SymmetricTensor{2,3,T}, @view x[1:6])
        R_ = x[7]
        X1_ = fromvoigt(SymmetricTensor{2,3,T}, @view x[8:13])
        X2_ = fromvoigt(SymmetricTensor{2,3,T}, @view x[14:19])
        #q_ = x[20]
        #zeta_ = fromvoigt(SymmetricTensor{2,3,T}, @view x[21:26])

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
    return g3!
    #return g2!
    #return g!
end
