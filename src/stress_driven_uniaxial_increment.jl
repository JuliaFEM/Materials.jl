# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

@with_kw mutable struct DSADriverState <: AbstractMaterialState
    time :: Float64 = zero(Float64)
    strain :: SymmetricTensor{2,3} = zero(SymmetricTensor{2,3,Float64})
end

@with_kw struct DSAParameterState <: AbstractMaterialState
    E :: Float64 = 0.0
    nu :: Float64 = 0.0
    R0 :: Float64 = 0.0
    Kn :: Float64 = 0.0
    nn :: Float64 = 0.0
    C1 :: Float64 = 0.0
    D1 :: Float64 = 0.0
    C2 :: Float64 = 0.0
    D2 :: Float64 = 0.0
    Q :: Float64 = 0.0
    b :: Float64 = 0.0
    w :: Float64 = 0.0
    P1 :: Float64 = 0.0
    P2 :: Float64 = 0.0
    m :: Float64 = 0.0
    ba :: Float64 = 0.0
    zeta :: Float64 = 0.0
end

@with_kw struct DSAVariableState <: AbstractMaterialState
    stress :: SymmetricTensor{2,3} = zero(SymmetricTensor{2,3,Float64})
    X1 :: SymmetricTensor{2,3} = zero(SymmetricTensor{2,3,Float64})
    X2 :: SymmetricTensor{2,3} = zero(SymmetricTensor{2,3,Float64})
    plastic_strain :: SymmetricTensor{2,3} = zero(SymmetricTensor{2,3,Float64})
    cumeq :: Float64 = zero(Float64)
    R :: Float64 = zero(Float64)
    jacobian :: SymmetricTensor{4,3} = zero(SymmetricTensor{4,3,Float64})
    ta :: Float64 = zero(Float64)
    Ra :: Float64 = zero(Float64)
end

@with_kw mutable struct DSA <: AbstractMaterial
    drivers :: DSADriverState = DSADriverState()
    ddrivers :: DSADriverState = DSADriverState()
    variables :: DSAVariableState = DSAVariableState()
    variables_new :: DSAVariableState = DSAVariableState()
    parameters :: DSAParameterState = DSAParameterState()
    dparameters :: DSAParameterState = DSAParameterState()
end

function integrate_material!(material::DSA)
    p = material.parameters
    v = material.variables
    dd = material.ddrivers
    d = material.drivers
    @unpack E, nu, R0, Kn, nn, C1, D1, C2, D2, Q, b, w, P1, P2, m, ba, zeta = p
    mu = E/(2.0*(1.0+nu))
    lambda = E*nu/((1.0+nu)*(1.0-2.0*nu))

    @unpack strain, time = d
    dstrain = dd.strain
    dtime = dd.time
    @unpack stress, X1, X2, plastic_strain, cumeq, R, jacobian, ta, Ra = v

    jacobian = isotropic_elasticity_tensor(lambda, mu)

    stress += dcontract(jacobian, dstrain)
    seff = stress - X1 - X2
    seff_dev = dev(seff)
    f = sqrt(1.5)*norm(seff_dev) - (R0 + R + (1 - zeta) * Ra)
    if f > 0.0
        g! = create_nonlinear_system_of_equations(material)
        x0 = [tovoigt(stress); R; tovoigt(X1); tovoigt(X2); ta + dtime; Ra]
        F = similar(x0)
        res = nlsolve(g!, x0; autodiff = :forward)
        x = res.zero
        res.f_converged || error("Nonlinear system of equations did not converge!")

        stress = fromvoigt(SymmetricTensor{2,3,Float64}, @view x[1:6])
        R = x[7]
        X1 = fromvoigt(SymmetricTensor{2,3,Float64}, @view x[8:13])
        X2 = fromvoigt(SymmetricTensor{2,3,Float64}, @view x[14:19])
        ta = x[20]
        Ra = x[21]
        seff = stress - X1 - X2
        seff_dev = dev(seff)
        f = sqrt(1.5)*norm(seff_dev) - (R0 + R + (1 - zeta) * Ra)
        dotp = ((f >= 0.0 ? f : 0.0)/(Kn + zeta * Ra))^nn
        dp = dotp*dtime
        n = sqrt(1.5)*seff_dev/norm(seff_dev)

        plastic_strain += dp*n
        cumeq += dp
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
    else
        ta += dtime    
    end 
    variables_new = DSAVariableState(stress = stress,
                                          X1 = X1,
                                          X2 = X2,
                                          R = R,
                                          plastic_strain = plastic_strain,
                                          cumeq = cumeq,
                                          jacobian = jacobian,
                                          ta = ta,
                                          Ra = Ra)
    material.variables_new = variables_new
end

function create_nonlinear_system_of_equations(material::DSA)
    p = material.parameters
    v = material.variables
    dd = material.ddrivers
    d = material.drivers
    @unpack E, nu, R0, Kn, nn, C1, D1, C2, D2, Q, b, w, P1, P2, m, ba, zeta = p
    mu = E/(2.0*(1.0+nu))
    lambda = E*nu/((1.0+nu)*(1.0-2.0*nu))

    @unpack strain, time = d
    dstrain = dd.strain
    dtime = dd.time
    @unpack stress, X1, X2, plastic_strain, cumeq, R, ta, Ra = v

    function g!(F, x::Vector{T}) where {T} # System of non-linear equations
        jacobian = isotropic_elasticity_tensor(lambda, mu)
        stress_ = fromvoigt(SymmetricTensor{2,3,T}, @view x[1:6])
        R_ = x[7]
        X1_ = fromvoigt(SymmetricTensor{2,3,T}, @view x[8:13])
        X2_ = fromvoigt(SymmetricTensor{2,3,T}, @view x[14:19])
        ta_ = x[20]
        Ra_ = x[21]

        seff = stress_ - X1_ - X2_
        seff_dev = dev(seff)
        f = sqrt(1.5)*norm(seff_dev) - (R0 + R_ + (1 - zeta) * Ra_)

        dotp = ((f >= 0.0 ? f : 0.0)/(Kn + zeta * Ra_))^nn
        dp = dotp*dtime
        n = sqrt(1.5)*seff_dev/norm(seff_dev)
        dstrain_plastic = dp*n
        Ras  = P1 * (1 - exp( -P2 * ta_^m ))
        tovoigt!(view(F, 1:6), stress - stress_ + dcontract(jacobian, dstrain - dstrain_plastic))
        F[7] = R - R_ + b*(Q-R_)*dp
        if isapprox(C1, 0.0)
            tovoigt!(view(F,8:13),X1 - X1_)
        else
            tovoigt!(view(F,8:13), X1 - X1_ + 2.0/3.0*C1*dp*(n - 1.5*D1/C1*X1_))
        end
        if isapprox(C2, 0.0)
            tovoigt!(view(F,14:19), X2 - X2_)
        else
            tovoigt!(view(F, 14:19), X2 - X2_ + 2.0/3.0*C2*dp*(n - 1.5*D2/C2*X2_))
        end

        F[20] = ta - ta_ + dtime - ta_/w * dp
        F[21] = Ra - Ra_ + b*(Ras-Ra_)*dp

        println(F)

    end
    return g!
end
