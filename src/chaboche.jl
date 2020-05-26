# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

# TODO: parametrize the short aliases on the element type?
typealias SymmT2 SymmetricTensor{2,3}
typealias SymmT4 SymmetricTensor{4,3}
typealias SymmT2Float64 SymmetricTensor{2,3,Float64}
typealias SymmT4Float64 SymmetricTensor{4,3,Float64}

@with_kw mutable struct ChabocheDriverState <: AbstractMaterialState
    time :: Float64 = zero(Float64)
    strain :: SymmT2 = zero(SymmT2Float64)
end

@with_kw struct ChabocheParameterState <: AbstractMaterialState
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
end

@with_kw struct ChabocheVariableState <: AbstractMaterialState
    stress :: SymmT2 = zero(SymmT2Float64)
    X1 :: SymmT2 = zero(SymmT2Float64)
    X2 :: SymmT2 = zero(SymmT2Float64)
    plastic_strain :: SymmT2 = zero(SymmT2Float64)
    cumeq :: Float64 = zero(Float64)
    R :: Float64 = zero(Float64)
    jacobian :: SymmT4 = zero(SymmT4Float64)
end

@with_kw mutable struct Chaboche <: AbstractMaterial
    drivers :: ChabocheDriverState = ChabocheDriverState()
    ddrivers :: ChabocheDriverState = ChabocheDriverState()
    variables :: ChabocheVariableState = ChabocheVariableState()
    variables_new :: ChabocheVariableState = ChabocheVariableState()
    parameters :: ChabocheParameterState = ChabocheParameterState()
    dparameters :: ChabocheParameterState = ChabocheParameterState()
end

# Convert the elastic constants (E, ν) to the Lamé constants (μ, λ). Isotropic material.
@inline function lame(E, ν)
    μ = E/(2.0*(1.0+nu))
    λ = E*nu/((1.0+nu)*(1.0-2.0*nu))
    return μ, λ
end

# Adaptors for NLsolve. Marshal the problem state into a Vector and back.
@inline function state_to_vector(σ::SymmT2Float64,
                                 R::Float64,
                                 X1::SymmT2Float64,
                                 X2::SymmT2Float64)
    return [tovoigt(σ), R, tovoigt(X1), tovoigt(X2)]
end
@inline function state_from_vector(x::Vector{Float64})
    σ = fromvoigt(SymmT2Float64, @view x[1:6])
    R = x[7]
    X1 = fromvoigt(SymmT2Float64, @view x[8:13])
    X2 = fromvoigt(SymmT2Float64, @view x[14:19])
    return σ, R, X1, X2
end

# Take away the bang; i.e. produce an equivalent single-argument non-mutating
# function f (that allocates and returns the result), when given a two-argument
# mutating function f! (which writes its result into the first argument).
#
# The argument and result types of F! must be the same.
#
# TODO: To generalize, we could make a @generated function that reflects on the
# parameter types of f!, and creates `out` using the type of f!'s first argument
# (instead of using `x` as we do now). Of course we're still relying on the type having
# a sensible default constructor (to perform the allocation when similar() is called).
function debang(f!)
    function f(x)
        out = similar(x)
        f!(out, x)
        return out
    end
    return f
end

function integrate_material!(material::Chaboche)
    p = material.parameters
    v = material.variables
    dd = material.ddrivers
    d = material.drivers
    @unpack E, nu, R0, Kn, nn, C1, D1, C2, D2, Q, b = p
    mu, lambda = lame(E, nu)

    @unpack strain, time = d
    dstrain = dd.strain
    dtime = dd.time
    @unpack stress, X1, X2, plastic_strain, cumeq, R, jacobian = v

    # elastic part
    jacobian = isotropic_elasticity_tensor(lambda, mu)  # dσ/dε, i.e. ∂σij/∂εkl
    stress += dcontract(jacobian, dstrain)  # add the elastic stress increment, get the intermediate stress

    # resulting deviatoric plastic stress (accounting for backstresses Xm)
    seff_dev = dev(stress - X1 - X2)
    # von Mises yield function; f := J(seff_dev) - Y
    f = sqrt(1.5)*norm(seff_dev) - (R0 + R)  # using intermediate problem state, after elastic update
    if f > 0.0
        g! = create_nonlinear_system_of_equations(material)
        x0 = state_to_vector(stress, R, X1, X2)
        res = nlsolve(g!, x0; autodiff = :forward)  # user manual: https://github.com/JuliaNLSolvers/NLsolve.jl
        converged(res) || error("Nonlinear system of equations did not converge!")
        x = res.zero
        stress, R, X1, X2 = state_from_vector(x)

        seff_dev = dev(stress - X1 - X2)
        f = sqrt(1.5)*norm(seff_dev) - (R0 + R)  # using new problem state

        dotp = ((f >= 0.0 ? f : 0.0)/Kn)^nn  # plasticity multiplier, see equations (3) and (4) in Chaboche 2013
        dp = dotp*dtime  # |dε_p|, using backward Euler (dotp is ∂ε_p/∂t at the end of the timestep)
        n = sqrt(1.5)*seff_dev/norm(seff_dev)  # Chaboche: a (tensorial) unit direction, s.t. 2/3 * (n : n) = 1

        plastic_strain += dp*n
        cumeq += dp   # TODO: Some kind of 1D cumulative plastic strain? (Note dp ≥ 0.) What is this used for?

        # Compute the new Jacobian, accounting for the plastic contribution. Because
        #   x ≡ [σ R X1 X2]   (vector of length 19, with tensors encoded in Voigt format)
        # we have
        #   (dx/dε)[1:6,1:6] = dσ/dε
        # for which we can compute the LHS as follows:
        #   dx/dε = dx/dr dr/dε = inv(dr/dx) dr/dε ≡ (dr/dx) \ (dr/dε)
        # where r = r(x) is the residual, given by the function g!. AD can get us dr/dx automatically,
        # the other factor we will have to supply manually.
        drdx = ForwardDiff.jacobian(debang(g!), x)  # Array{19, 19}
        drde = zeros((length(x),6))                 # Array{19, 6}
        # We are only interested in dσ/dε, so the rest of drde can be left as zeros.
        # TODO: where does the minus sign come from?
        drde[1:6, 1:6] = -tovoigt(jacobian)  # (negative of the) elastic Jacobian. Follows from the defn. of g!.
        jacobian = fromvoigt(SymmT4, (drdx\drde)[1:6, 1:6])
    end
    variables_new = ChabocheVariableState(stress = stress,
                                          X1 = X1,
                                          X2 = X2,
                                          R = R,
                                          plastic_strain = plastic_strain,
                                          cumeq = cumeq,
                                          jacobian = jacobian)
    material.variables_new = variables_new
end

function create_nonlinear_system_of_equations(material::Chaboche)
    p = material.parameters
    v = material.variables
    dd = material.ddrivers
    d = material.drivers
    @unpack E, nu, R0, Kn, nn, C1, D1, C2, D2, Q, b = p
    mu, lambda = lame(E, nu)

    # Store the old problem state (i.e. the problem state at the time when this
    # equation system instance was created).
    @unpack strain, time = d
    dstrain = dd.strain
    dtime = dd.time
    @unpack stress, X1, X2, plastic_strain, cumeq, R = v

    # Compute the residual. F is output, x is filled by NLsolve.
    # The solution is x = x* such that g(x*) = 0.
    function g!(F, x::Vector{T}) where {T}
        jacobian = isotropic_elasticity_tensor(lambda, mu)  # TODO: why not compute once, store in closure?
        stress_, R_, X1_, X2_ = state_from_vector(x)  # tentative new values from nlsolve

        seff_dev = dev(stress_ - X1_ - X2_)
        f = sqrt(1.5)*norm(seff_dev) - (R0 + R_)

        dotp = ((f >= 0.0 ? f : 0.0)/Kn)^nn
        dp = dotp*dtime
        n = sqrt(1.5)*seff_dev/norm(seff_dev)

        # The equations are written in a delta form:
        #
        # Δσ = (∂σ/∂ε)_e : dε_e = (∂σ/∂ε)_e : (dε - dε_p)   (components 1:6)
        # ΔR = b (Q - R_new) |dε|                           (component 7)
        # ΔX1 = (2/3) C1 |dε| (n - (3/2) (D1/C1) X1_new)    (components 8:13)
        # ΔX2 = (2/3) C2 |dε| (n - (3/2) (D2/C2) X2_new)    (components 14:19)
        #
        # where
        #
        # Δ(...) = (...)_new - (...)_old
        #
        # Then move the delta terms to the RHS to get the standard form, (stuff) = 0.
        #
        dstrain_plastic = dp*n
        dstrain_elastic = dstrain - dstrain_plastic
        tovoigt!(view(F, 1:6), stress - stress_ + dcontract(jacobian, dstrain_elastic))
        F[7] = R - R_ + b*(Q - R_)*dp
        if isapprox(C1, 0.0)
            # TODO: In order for the below general case to reduce to this, both C1 **and D1** must be zero.
            tovoigt!(view(F, 8:13), X1 - X1_)
        else
            tovoigt!(view(F, 8:13), X1 - X1_ + 2.0/3.0*C1*dp*(n - 1.5*D1/C1*X1_))
        end
        if isapprox(C2, 0.0)
            # TODO: In order for the below general case to reduce to this, both C2 **and D2** must be zero.
            tovoigt!(view(F, 14:19), X2 - X2_)
        else
            tovoigt!(view(F, 14:19), X2 - X2_ + 2.0/3.0*C2*dp*(n - 1.5*D2/C2*X2_))
        end
    end
    return g!
end
