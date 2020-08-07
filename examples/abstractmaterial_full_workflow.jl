using Tensors
using Parameters
using NLsolve
using ForwardDiff
using Profile

abstract type AbstractMaterial end
abstract type AbstractMaterialState end

@generated function Base.:+(state::T, dstate::T) where {T <: AbstractMaterialState}
   expr = [:(state.$p+ dstate.$p) for p in fieldnames(T)]
   return :(T($(expr...)))
end

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

@with_kw mutable struct PerfectPlasticDriverState <: AbstractMaterialState
    time :: Float64 = zero(Float64)
    strain :: Symm2 = zero(Symm2{Float64})
end

@with_kw struct PerfectPlasticParameterState <: AbstractMaterialState
    youngs_modulus :: Float64 = zero(Float64)
    poissons_ratio :: Float64 = zero(Float64)
    yield_stress :: Float64 = zero(Float64)
end

@with_kw struct PerfectPlasticVariableState <: AbstractMaterialState
    stress :: Symm2 = zero(Symm2{Float64})
    plastic_strain :: Symm2 = zero(Symm2{Float64})
    cumeq :: Float64 = zero(Float64)
end

function update!(material::M) where {M <: AbstractMaterial}
    material.drivers += material.ddrivers
    material.parameters += material.dparameters
    material.variables = material.variables_new
    # material.ddrivers = typeof(material.ddrivers)()
    # material.dparameters = typeof(material.dparameters)()
    # material.variables_new = typeof(material.variables_new)()
    reset!(material)
end

function reset!(material::M) where {M <: AbstractMaterial}
    material.ddrivers = typeof(material.ddrivers)()
    material.dparameters = typeof(material.dparameters)()
    material.variables_new = typeof(material.variables_new)()
end

# @with_kw mutable struct Material{M <: AbstractMaterial}
#     drivers :: DriverState{M} = DriverState{M}()
#     ddrivers :: DriverState{M} = DriverState{M}()
#     variables :: VariableState{M} = VariableState{M}()
#     dvariables :: VariableState{M} = VariableState{M}()
#     parameters :: ParameterState{M} = ParameterState{M}()
#     dparameters :: ParameterState{M} = ParameterState{M}()
# end

@with_kw mutable struct PerfectPlastic <: AbstractMaterial
    drivers :: PerfectPlasticDriverState = PerfectPlasticDriverState()
    ddrivers :: PerfectPlasticDriverState = PerfectPlasticDriverState()
    variables :: PerfectPlasticVariableState = PerfectPlasticVariableState()
    variables_new :: PerfectPlasticVariableState = PerfectPlasticVariableState()
    parameters :: PerfectPlasticParameterState = PerfectPlasticParameterState()
    dparameters :: PerfectPlasticParameterState = PerfectPlasticParameterState()
end


mat = Chaboche()
mat2 = PerfectPlastic()

function isotropic_elasticity_tensor(lambda, mu)
    delta(i,j) = i==j ? 1.0 : 0.0
    g(i,j,k,l) = lambda*delta(i,j)*delta(k,l) + mu*(delta(i,k)*delta(j,l)+delta(i,l)*delta(j,k))
    jacobian = Symm4{Float64}(g)
    return jacobian
end

function integrate_material!(material::Chaboche)
    p = material.parameters
    v = material.variables
    dd = material.ddrivers
    d = material.drivers
    @unpack E, nu, R0, Kn, nn, C1, D1, C2, D2, Q, b = p
    mu = E/(2.0*(1.0+nu))
    lambda = E*nu/((1.0+nu)*(1.0-2.0*nu))

    @unpack strain, time = d
    dstrain = dd.strain
    dtime = dd.time
    @unpack stress, X1, X2, plastic_strain, cumeq, R, jacobian = v

    jacobian = isotropic_elasticity_tensor(lambda, mu)

    stress += dcontract(jacobian, dstrain)
    seff = stress - X1 - X2
    seff_dev = dev(seff)
    f = sqrt(1.5)*norm(seff_dev) - (R0 + R)
    if f > 0.0
        g! = create_nonlinear_system_of_equations(material)
        x0 = [tovoigt(stress); R; tovoigt(X1); tovoigt(X2)]
        F = similar(x0)
        res = nlsolve(g!, x0; autodiff = :forward)
        x = res.zero
        res.f_converged || error("Nonlinear system of equations did not converge!")

        stress = fromvoigt(Symm2{Float64}, @view x[1:6])
        R = x[7]
        X1 = fromvoigt(Symm2{Float64}, @view x[8:13])
        X2 = fromvoigt(Symm2{Float64}, @view x[14:19])
        seff = stress - X1 - X2
        seff_dev = dev(seff)
        f = sqrt(1.5)*norm(seff_dev) - (R0 + R)
        dotp = ((f >= 0.0 ? f : 0.0)/Kn)^nn
        dp = dotp*dtime
        n = 1.5*seff_dev/norm(seff_dev)

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

function create_nonlinear_system_of_equations(material::Chaboche)
    p = material.parameters
    v = material.variables
    dd = material.ddrivers
    d = material.drivers
    @unpack E, nu, R0, Kn, nn, C1, D1, C2, D2, Q, b = p
    mu = E/(2.0*(1.0+nu))
    lambda = E*nu/((1.0+nu)*(1.0-2.0*nu))

    @unpack strain, time = d
    dstrain = dd.strain
    dtime = dd.time
    @unpack stress, X1, X2, plastic_strain, cumeq, R = v

    function g!(F, x::Vector{T}) where {T} # System of non-linear equations
        jacobian = isotropic_elasticity_tensor(lambda, mu)
        stress_ = fromvoigt(Symm2{T}, @view x[1:6])
        R_ = x[7]
        X1_ = fromvoigt(Symm2{T}, @view x[8:13])
        X2_ = fromvoigt(Symm2{T}, @view x[14:19])

        seff = stress_ - X1_ - X2_
        seff_dev = dev(seff)
        f = sqrt(1.5)*norm(seff_dev) - (R0 + R_)

        dotp = ((f >= 0.0 ? f : 0.0)/Kn)^nn
        dp = dotp*dtime
        n = 1.5*seff_dev/norm(seff_dev)
        dstrain_plastic = dp*n
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
    end
    return g!
end

function simple_integration_test()
    parameters = ChabocheParameterState(E = 200.0e3,
                                        nu = 0.3,
                                        R0 = 100.0,
                                        Kn = 100.0,
                                        nn = 10.0,
                                        C1 = 10000.0,
                                        D1 = 100.0,
                                        C2 = 50000.0,
                                        D2 = 1000.0,
                                        Q = 50.0,
                                        b = 0.1)

    dstrain_dtime = fromvoigt(Symm2{Float64}, 1e-3*[1.0, -0.3, -0.3, 0.0, 0.0, 0.0]; offdiagscale=2.0)
    ddrivers = ChabocheDriverState(time = 0.25, strain = 0.25*dstrain_dtime)
    chabmat = Chaboche(parameters = parameters, ddrivers = ddrivers)
    @info "time = $(chabmat.drivers.time), stress = $(chabmat.variables.stress)"
    integrate_material!(chabmat)
    update!(chabmat)
    @info "time = $(chabmat.drivers.time), stress = $(chabmat.variables.stress)"

    chabmat.ddrivers = ddrivers
    integrate_material!(chabmat)
    update!(chabmat)
    @info "time = $(chabmat.drivers.time), stress = $(chabmat.variables.stress)"

    chabmat.ddrivers = ddrivers
    integrate_material!(chabmat)
    update!(chabmat)
    @info "time = $(chabmat.drivers.time), stress = $(chabmat.variables.stress)"
end

using DelimitedFiles, Test
path = joinpath(@__DIR__, "one_elem_disp_chaboche", "unitelement_results.rpt")
data = readdlm(path, Float64; skipstart=4)
ts = data[:,1]
s11_ = data[:,2]
s12_ = data[:,3]
s13_ = data[:,4]
s22_ = data[:,5]
s23_ = data[:,6]
s33_ = data[:,7]
e11_ = data[:,8]
e12_ = data[:,9]
e13_ = data[:,10]
e22_ = data[:,11]
e23_ = data[:,12]
e33_ = data[:,13]

strains = [[e11_[i], e22_[i], e33_[i], e23_[i], e13_[i], e12_[i]] for i in 1:length(ts)]

function test_chaboche()
    parameters = ChabocheParameterState(E = 200.0e3,
                                        nu = 0.3,
                                        R0 = 100.0,
                                        Kn = 100.0,
                                        nn = 10.0,
                                        C1 = 10000.0,
                                        D1 = 100.0,
                                        C2 = 50000.0,
                                        D2 = 1000.0,
                                        Q = 50.0,
                                        b = 0.1)
    chabmat = Chaboche(parameters = parameters)
    s33s = [chabmat.variables.stress[3,3]]
    for i=2:length(ts)
        dtime = ts[i]-ts[i-1]
        dstrain = fromvoigt(Symm2{Float64}, strains[i]-strains[i-1]; offdiagscale=2.0)
        chabmat.ddrivers = ChabocheDriverState(time = dtime, strain = dstrain)
        integrate_material!(chabmat)
        update!(chabmat)
        push!(s33s, chabmat.variables.stress[3,3])
    end
    # @test isapprox(s33s, s33_; rtol=0.01)
end

test_chaboche()
Profile.clear_malloc_data()
test_chaboche()
# using BenchmarkTools
# @btime test_chaboche()

function simple_integration_test_fd_tangent()
    parameters = ChabocheParameterState(E = 200.0e3,
                                        nu = 0.3,
                                        R0 = 100.0,
                                        Kn = 100.0,
                                        nn = 10.0,
                                        C1 = 10000.0,
                                        D1 = 100.0,
                                        C2 = 50000.0,
                                        D2 = 1000.0,
                                        Q = 50.0,
                                        b = 0.1)

    dstrain_dtime = fromvoigt(Symm2{Float64}, 1e-3*[1.0, -0.3, -0.3, 0.0, 0.0, 0.0]; offdiagscale=2.0)
    ddrivers = ChabocheDriverState(time = 0.25, strain = 0.25*dstrain_dtime)
    chabmat = Chaboche(parameters = parameters, ddrivers = ddrivers)

    function get_stress(dstrain)
        chabmat.ddrivers.strain = dstrain
        integrate_material!(chabmat)
        return chabmat.variables_new.stress
    end
    # stress = get_stress(0.25*dstrain_dtime)
    # @info "stress = $stress"

    D, dstress = Tensors.gradient(get_stress, 0.25*dstrain_dtime, :all)
    @info "D_mat = $(tovoigt(chabmat.variables_new.jacobian))"
    @info "D = $(tovoigt(D))"

    chabmat.variables_new = typeof(chabmat.variables_new)()
    chabmat.ddrivers = ChabocheDriverState(time = 0.25, strain = 0.25*dstrain_dtime)
    @info "time = $(chabmat.drivers.time), stress = $(chabmat.variables.stress)"
    integrate_material!(chabmat)
    update!(chabmat)
    @info "time = $(chabmat.drivers.time), stress = $(chabmat.variables.stress)"

    function get_stress(dstrain)
        chabmat.ddrivers.strain = dstrain
        integrate_material!(chabmat)
        return chabmat.variables_new.stress
    end
    # stress = get_stress(0.25*dstrain_dtime)
    # @info "stress = $stress"

    D, dstress = Tensors.gradient(get_stress, 0.25*dstrain_dtime, :all)
    @info "D = $(tovoigt(D))"
    chabmat.variables_new = typeof(chabmat.variables_new)()
    chabmat.ddrivers = ChabocheDriverState(time = 0.25, strain = 0.25*dstrain_dtime)
    integrate_material!(chabmat)
    update!(chabmat)
    @info "time = $(chabmat.drivers.time), stress = $(chabmat.variables.stress)"
    function get_stress(dstrain)
        chabmat.ddrivers.strain = dstrain
        integrate_material!(chabmat)
        return chabmat.variables_new.stress
    end
    # stress = get_stress(0.25*dstrain_dtime)
    # @info "stress = $stress"

    D, dstress = Tensors.gradient(get_stress, 0.25*dstrain_dtime, :all)
    @info "D = $(tovoigt(D))"
    chabmat.variables_new = typeof(chabmat.variables_new)()
    chabmat.ddrivers = ChabocheDriverState(time = 0.25, strain = 0.25*dstrain_dtime)
    integrate_material!(chabmat)
    update!(chabmat)
    @info "time = $(chabmat.drivers.time), stress = $(chabmat.variables.stress)"
    function get_stress(dstrain)
        chabmat.ddrivers.strain = dstrain
        integrate_material!(chabmat)
        return chabmat.variables_new.stress
    end
    # stress = get_stress(0.25*dstrain_dtime)
    # @info "stress = $stress"

    D, dstress = Tensors.gradient(get_stress, 0.25*dstrain_dtime, :all)
    @info "D = $(tovoigt(D))"
end

# simple_integration_test_fd_tangent()

function simple_integration_test_fd_tangent2()
    parameters = ChabocheParameterState(E = 200.0e3,
                                        nu = 0.3,
                                        R0 = 100.0,
                                        Kn = 100.0,
                                        nn = 10.0,
                                        C1 = 10000.0,
                                        D1 = 100.0,
                                        C2 = 50000.0,
                                        D2 = 1000.0,
                                        Q = 50.0,
                                        b = 0.1)

    dstrain_dtime = fromvoigt(Symm2{Float64}, 1e-3*[1.0, -0.3, -0.3, 0.0, 0.0, 0.0]; offdiagscale=2.0)
    ddrivers = ChabocheDriverState(time = 0.25, strain = 0.25*dstrain_dtime)
    chabmat = Chaboche(parameters = parameters, ddrivers = ddrivers)
    integrate_material!(chabmat)
    update!(chabmat)
    @info "time = $(chabmat.drivers.time), stress = $(chabmat.variables.stress)"

    chabmat.ddrivers = ddrivers
    integrate_material!(chabmat)
    update!(chabmat)
    @info "time = $(chabmat.drivers.time), stress = $(chabmat.variables.stress)"

    chabmat.ddrivers = ddrivers
    integrate_material!(chabmat)
    g! = create_nonlinear_system_of_equations(chabmat)
    function residuals(x)
        F = similar(x)
        g!(F, x)
        return F
    end

    x0 = [tovoigt(chabmat.variables_new.stress); chabmat.variables_new.R; tovoigt(chabmat.variables_new.X1); tovoigt(chabmat.variables_new.X2)]
    drdx = ForwardDiff.jacobian(residuals, x0)
    @info "size(drdx) = $(size(drdx))"
    @info "drdx = $drdx"
    @unpack E, nu, R0, Kn, nn, C1, D1, C2, D2, Q, b = parameters
    mu = E/(2.0*(1.0+nu))
    lambda = E*nu/((1.0+nu)*(1.0-2.0*nu))
    jacobian = isotropic_elasticity_tensor(lambda, mu)
    drde = zeros((19,6))
    drde[1:6, 1:6] = -tovoigt(jacobian)
    @info "drde = $drde"
    @info "size(drde) = $(size(drde))"
    jacobian2 = (drdx\drde)[1:6, 1:6]
    @info "jacobian = $(tovoigt(jacobian))"
    @info "jacobian2 = $jacobian2"
    jacobian3 = (drdx[1:6, 1:6] +  drdx[1:6,7:end]*(drdx[7:end,7:end]\-drdx[7:end, 1:6]))\drde[1:6, 1:6]
    @info "jacobian3 = $jacobian3"
    @info "jacobian4 = $(tovoigt(chabmat.variables_new.jacobian))"
    update!(chabmat)
    @info "time = $(chabmat.drivers.time), stress = $(chabmat.variables.stress)"

    chabmat.ddrivers = ddrivers
    integrate_material!(chabmat)
    g! = create_nonlinear_system_of_equations(chabmat)
    function residuals(x)
        F = similar(x)
        g!(F, x)
        return F
    end

    x0 = [tovoigt(chabmat.variables_new.stress); chabmat.variables_new.R; tovoigt(chabmat.variables_new.X1); tovoigt(chabmat.variables_new.X2)]
    drdx = ForwardDiff.jacobian(residuals, x0)
    @info "size(drdx) = $(size(drdx))"
    @info "drdx = $drdx"
    @unpack E, nu, R0, Kn, nn, C1, D1, C2, D2, Q, b = parameters
    mu = E/(2.0*(1.0+nu))
    lambda = E*nu/((1.0+nu)*(1.0-2.0*nu))
    jacobian = isotropic_elasticity_tensor(lambda, mu)
    drde = zeros((19,6))
    drde[1:6, 1:6] = -tovoigt(jacobian)
    @info "drde = $drde"
    @info "size(drde) = $(size(drde))"
    jacobian2 = (drdx\drde)[1:6, 1:6]
    @info "jacobian = $(tovoigt(jacobian))"
    @info "jacobian2 = $jacobian2"
    jacobian3 = (drdx[1:6, 1:6] +  drdx[1:6,7:end]*(drdx[7:end,7:end]\-drdx[7:end, 1:6]))\drde[1:6, 1:6]
    @info "jacobian3 = $jacobian3"
    @info "jacobian4 = $(tovoigt(chabmat.variables_new.jacobian))"
    update!(chabmat)
    @info "time = $(chabmat.drivers.time), stress = $(chabmat.variables.stress)"
end
#simple_integration_test_fd_tangent2()
