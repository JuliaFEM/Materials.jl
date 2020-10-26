# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE
#
# Some examples of how to use the Chaboche material model.

using Parameters
using ForwardDiff
using DelimitedFiles, Test

using Materials

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
    ddrivers = ChabocheDriverState(time=0.25, strain=0.25*dstrain_dtime)
    chabmat = Chaboche(parameters=parameters, ddrivers=ddrivers)
    @info "time = $(chabmat.drivers.time), stress = $(chabmat.variables.stress)"

    integrate_material!(chabmat)
    update_material!(chabmat)
    @info "time = $(chabmat.drivers.time), stress = $(chabmat.variables.stress)"

    chabmat.ddrivers = ddrivers
    integrate_material!(chabmat)
    update_material!(chabmat)
    @info "time = $(chabmat.drivers.time), stress = $(chabmat.variables.stress)"

    chabmat.ddrivers = ddrivers
    integrate_material!(chabmat)
    update_material!(chabmat)
    @info "time = $(chabmat.drivers.time), stress = $(chabmat.variables.stress)"
end
simple_integration_test()

function test_chaboche()
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
    chabmat = Chaboche(parameters=parameters)
    s33s = [chabmat.variables.stress[3,3]]
    for i=2:length(ts)
        dtime = ts[i]-ts[i-1]
        dstrain = fromvoigt(Symm2{Float64}, strains[i]-strains[i-1]; offdiagscale=2.0)
        chabmat.ddrivers = ChabocheDriverState(time = dtime, strain = dstrain)
        integrate_material!(chabmat)
        update_material!(chabmat)
        push!(s33s, chabmat.variables.stress[3,3])
    end
    @test isapprox(s33s, s33_; rtol=0.01)
end
test_chaboche()

# Profile.clear_malloc_data()
# test_chaboche()
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
    ddrivers = ChabocheDriverState(time=0.25, strain=0.25*dstrain_dtime)
    chabmat = Chaboche(parameters=parameters, ddrivers=ddrivers)

    function get_stress(dstrain::Symm2)
        chabmat.ddrivers.strain = dstrain
        integrate_material!(chabmat)
        return chabmat.variables_new.stress
    end

    # stress = get_stress(0.25*dstrain_dtime)
    # @info "stress = $stress"

    # https://kristofferc.github.io/Tensors.jl/stable/man/automatic_differentiation/
    # TODO: doesn't work, a Nothing ends up in the type for some reason?
    D, dstress = Tensors.gradient(get_stress, 0.25*dstrain_dtime, :all)
    @info "D_mat = $(tovoigt(chabmat.variables_new.jacobian))"
    @info "D = $(tovoigt(D))"

    chabmat.variables_new = typeof(chabmat.variables_new)()
    chabmat.ddrivers = ChabocheDriverState(time=0.25, strain=0.25*dstrain_dtime)
    @info "time = $(chabmat.drivers.time), stress = $(chabmat.variables.stress)"
    integrate_material!(chabmat)
    update_material!(chabmat)
    @info "time = $(chabmat.drivers.time), stress = $(chabmat.variables.stress)"

    # stress = get_stress(0.25*dstrain_dtime)
    # @info "stress = $stress"

    D, dstress = Tensors.gradient(get_stress, 0.25*dstrain_dtime, :all)
    @info "D = $(tovoigt(D))"
    chabmat.variables_new = typeof(chabmat.variables_new)()
    chabmat.ddrivers = ChabocheDriverState(time=0.25, strain=0.25*dstrain_dtime)
    integrate_material!(chabmat)
    update_material!(chabmat)
    @info "time = $(chabmat.drivers.time), stress = $(chabmat.variables.stress)"

    # stress = get_stress(0.25*dstrain_dtime)
    # @info "stress = $stress"

    D, dstress = Tensors.gradient(get_stress, 0.25*dstrain_dtime, :all)
    @info "D = $(tovoigt(D))"
    chabmat.variables_new = typeof(chabmat.variables_new)()
    chabmat.ddrivers = ChabocheDriverState(time=0.25, strain=0.25*dstrain_dtime)
    integrate_material!(chabmat)
    update_material!(chabmat)
    @info "time = $(chabmat.drivers.time), stress = $(chabmat.variables.stress)"

    # stress = get_stress(0.25*dstrain_dtime)
    # @info "stress = $stress"

    D, dstress = Tensors.gradient(get_stress, 0.25*dstrain_dtime, :all)
    @info "D = $(tovoigt(D))"
end
simple_integration_test_fd_tangent()

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
    ddrivers = ChabocheDriverState(time=0.25, strain=0.25*dstrain_dtime)
    chabmat = Chaboche(parameters=parameters, ddrivers=ddrivers)
    integrate_material!(chabmat)
    update_material!(chabmat)
    @info "time = $(chabmat.drivers.time), stress = $(chabmat.variables.stress)"

    chabmat.ddrivers = ddrivers
    integrate_material!(chabmat)
    update_material!(chabmat)
    @info "time = $(chabmat.drivers.time), stress = $(chabmat.variables.stress)"

    chabmat.ddrivers = ddrivers
    integrate_material!(chabmat)
    g! = Materials.ChabocheModule.create_nonlinear_system_of_equations(chabmat)

    x0 = [tovoigt(chabmat.variables_new.stress); chabmat.variables_new.R; tovoigt(chabmat.variables_new.X1); tovoigt(chabmat.variables_new.X2)]
    drdx = ForwardDiff.jacobian(debang(g!), x0)
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
    update_material!(chabmat)
    @info "time = $(chabmat.drivers.time), stress = $(chabmat.variables.stress)"

    chabmat.ddrivers = ddrivers
    integrate_material!(chabmat)
    g! = Materials.ChabocheModule.create_nonlinear_system_of_equations(chabmat)

    x0 = [tovoigt(chabmat.variables_new.stress); chabmat.variables_new.R; tovoigt(chabmat.variables_new.X1); tovoigt(chabmat.variables_new.X2)]
    drdx = ForwardDiff.jacobian(debang(g!), x0)
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
    update_material!(chabmat)
    @info "time = $(chabmat.drivers.time), stress = $(chabmat.variables.stress)"
end
simple_integration_test_fd_tangent2()
