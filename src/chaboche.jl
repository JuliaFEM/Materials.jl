# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

@with_kw mutable struct ChabocheDriverState <: AbstractMaterialState
    time :: Float64 = zero(Float64)
    strain :: Symm2 = zero(Symm2{Float64})
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
    stress :: Symm2 = zero(Symm2{Float64})
    X1 :: Symm2 = zero(Symm2{Float64})
    X2 :: Symm2 = zero(Symm2{Float64})
    plastic_strain :: Symm2 = zero(Symm2{Float64})
    cumeq :: Float64 = zero(Float64)
    R :: Float64 = zero(Float64)
    jacobian :: Symm4 = zero(Symm4{Float64})
end

@with_kw mutable struct Chaboche <: AbstractMaterial
    drivers :: ChabocheDriverState = ChabocheDriverState()
    ddrivers :: ChabocheDriverState = ChabocheDriverState()
    variables :: ChabocheVariableState = ChabocheVariableState()
    variables_new :: ChabocheVariableState = ChabocheVariableState()
    parameters :: ChabocheParameterState = ChabocheParameterState()
    dparameters :: ChabocheParameterState = ChabocheParameterState()
end

"""
    state_to_vector(σ::T, R::S, X1::T, X2::T) where T <: Symm2{S} where S <: Real

Marshal the problem state into a `Vector`. Adaptor for `nlsolve`.
"""
@inline function state_to_vector(σ::T, R::S, X1::T, X2::T) where T <: Symm2{S} where S <: Real
    return [tovoigt(σ), R, tovoigt(X1), tovoigt(X2)]
end

"""
    state_from_vector(x::AbstractVector{Real})

Unmarshal the problem state from a `Vector`. Adaptor for `nlsolve`.
"""
@inline function state_from_vector(x::AbstractVector{Real})
    σ = fromvoigt(Symm2{S}, @view x[1:6])
    R = x[7]
    X1 = fromvoigt(Symm2{S}, @view x[8:13])
    X2 = fromvoigt(Symm2{S}, @view x[14:19])
    return σ, R, X1, X2
end

function integrate_material!(material::Chaboche)
    p = material.parameters
    v = material.variables
    dd = material.ddrivers
    d = material.drivers
    @unpack E, nu, R0, Kn, nn, C1, D1, C2, D2, Q, b = p
    lambda, mu = lame(E, nu)

    @unpack strain, time = d
    dstrain = dd.strain
    dtime = dd.time
    @unpack stress, X1, X2, plastic_strain, cumeq, R = v

    # elastic part
    jacobian = isotropic_elasticity_tensor(lambda, mu)  # dσ/dε, i.e. ∂σij/∂εkl
    stress += dcontract(jacobian, dstrain)  # add the elastic stress increment, get the elastic trial stress

    # resulting deviatoric plastic stress (accounting for backstresses Xm)
    seff_dev = dev(stress - X1 - X2)
    # von Mises yield function; f := J(seff_dev) - Y
    f = sqrt(1.5)*norm(seff_dev) - (R0 + R)  # using elastic trial problem state
    if f > 0.0
        g! = create_nonlinear_system_of_equations(material)
        x0 = state_to_vector(stress, R, X1, X2)
        res = nlsolve(g!, x0; autodiff = :forward)  # user manual: https://github.com/JuliaNLSolvers/NLsolve.jl
        converged(res) || error("Nonlinear system of equations did not converge!")
        x = res.zero
        stress, R, X1, X2 = state_from_vector(x)

        # using the new problem state
        seff_dev = dev(stress - X1 - X2)
        f = sqrt(1.5)*norm(seff_dev) - (R0 + R)

        dotp = ((f >= 0.0 ? f : 0.0)/Kn)^nn  # plasticity multiplier, see equations (3) and (4) in Chaboche 2013
        dp = dotp*dtime  # |dε_p|, using backward Euler (dotp is ∂ε_p/∂t at the end of the timestep)
        n = sqrt(1.5)*seff_dev/norm(seff_dev)  # Chaboche: a (tensorial) unit direction, s.t. 2/3 * (n : n) = 1

        plastic_strain += dp*n
        cumeq += dp   # cumulative equivalent plastic strain (note dp ≥ 0)

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
        jacobian = fromvoigt(Symm4, (drdx\drde)[1:6, 1:6])
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

"""
    create_nonlinear_system_of_equations(material::Chaboche)

Create and return an instance of the equation system for the delta form of the
evolution equations of the Chaboche material.

Used internally for computing the plastic contribution in `integrate_material!`.

The input `material` represents the problem state at the end of the previous
timestep. The created equation system will hold its own copy of that state.

The equation system is represented as a mutating function `g!` that computes the
residual:

```julia
    g!(F::V, x::V) where V <: AbstractVector{Real}
```

Both `F` (output) and `x` (input) are length-19 vectors containing
[σ, R, X1, X2], in that order. The tensor quantities σ, X1, X2 are
encoded in Voigt format.

The function `g!` is intended to be handed over to `nlsolve`.
"""
function create_nonlinear_system_of_equations(material::Chaboche)
    p = material.parameters
    v = material.variables
    dd = material.ddrivers
    d = material.drivers
    @unpack E, nu, R0, Kn, nn, C1, D1, C2, D2, Q, b = p
    lambda, mu = lame(E, nu)

    # Old problem state (i.e. the problem state at the time when this equation
    # system instance was created).
    #
    # Note this does not include the elastic trial; this is the state at the
    # end of the previous timestep.
    @unpack strain, time = d
    dstrain = dd.strain
    dtime = dd.time
    @unpack stress, X1, X2, plastic_strain, cumeq, R = v

    jacobian = isotropic_elasticity_tensor(lambda, mu)

    # Compute the residual. F is output, x is filled by NLsolve.
    # The solution is x = x* such that g(x*) = 0.
    function g!(F::V, x::V) where V <: AbstractVector{Real}
        stress_, R_, X1_, X2_ = state_from_vector(x)  # tentative new values from nlsolve

        seff_dev = dev(stress_ - X1_ - X2_)
        f = sqrt(1.5)*norm(seff_dev) - (R0 + R_)

        dotp = ((f >= 0.0 ? f : 0.0)/Kn)^nn
        dp = dotp*dtime
        n = sqrt(1.5)*seff_dev/norm(seff_dev)

        # The equations are written in a delta form:
        #
        # Δσ = (∂σ/∂ε)_e : dε_e = (∂σ/∂ε)_e : (dε - dε_p)   (components 1:6)
        # ΔR = b (Q - R_new) |dε_p|                         (component 7)
        # ΔX1 = (2/3) C1 |dε_p| (n - (3/2) (D1/C1) X1_new)  (components 8:13)
        # ΔX2 = (2/3) C2 |dε_p| (n - (3/2) (D2/C2) X2_new)  (components 14:19)
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
        # dp is a scalar, so it commutes in multiplication. This allows us to avoid the division by C1.
        tovoigt!(view(F,  8:13), X1 + (-1.0 + dp*(2.0/3.0*C1*n - D1*X1_)))
        tovoigt!(view(F, 14:19), X2 + (-1.0 + dp*(2.0/3.0*C2*n - D2*X2_)))
    end
    return g!
end
