# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Test, Tensors

let dtime = 0.25,
    parameters = ChabocheParameterState(E=200.0e3,
                                        nu=0.3,
                                        R0=100.0,
                                        Kn=100.0,
                                        nn=10.0,
                                        C1=10000.0,
                                        D1=100.0,
                                        C2=50000.0,
                                        D2=1000.0,
                                        Q=50.0,
                                        b=0.1),
    mat = Chaboche{Float64}(parameters = parameters),
    dstrain11 = 1e-3*dtime,
    dstrain12 = 1e-3*dtime,
    dtimes = [dtime, dtime, dtime, dtime, 1.0],
    dstrains11 = dstrain11*[1.0, 1.0, 1.0, -1.0, -4.0],
    dstrains12 = dstrain12*[1.0, 1.0, 1.0, -1.0, -4.0]

    plastic_flow_occurred = zeros(Bool, length(dstrains11) - 1)
    for i in 1:length(dtimes)
        dstrain11 = dstrains11[i]
        dstrain12 = dstrains12[i]
        dtime = dtimes[i]
        # TODO: After i > 1, never converges to within 1e-9. Figure out why.
        biaxial_increment!(mat, dstrain11, dstrain12, dtime; norm_acc=1e-8)
        update_material!(mat)
        if i > 1
            plastic_flow_occurred[i-1] = (mat.variables.cumeq > 0.0)
        end
        @test !iszero(mat.variables.stress[1,1]) && !iszero(mat.variables.stress[1,2])
        @test isapprox(tovoigt(mat.variables.stress; offdiagscale=2.0)[2:5], zeros(4); atol=1e-5)
    end
    @test any(plastic_flow_occurred)
end
