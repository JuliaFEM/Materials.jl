# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE
using Test, Tensors
dtime = 0.25
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
mat = Chaboche(parameters = parameters)
times = [mat.drivers.time]
stresses = [copy(tovoigt(mat.variables.stress))]
dstrain11 = 1e-3*dtime
dstrain12 = 1e-3*dtime

dtimes = [dtime, dtime, dtime, dtime, 1.0]
dstrains11 = [dstrain11, dstrain11, dstrain11, -dstrain11, -4*dstrain11]
dstrains12 = [dstrain12, dstrain12, dstrain12, -dstrain12, -4*dstrain12]
plasticity_test = zeros(length(dstrains11)-1)
for i in 1:length(dtimes)
    dstrain11 = dstrains11[i]
    dstrain12 = dstrains12[i]
    dtime = dtimes[i]
    biaxial_increment!(mat, dstrain11, dstrain12, dtime)
    update_material!(mat)
    push!(stresses, copy(tovoigt(mat.variables.stress)))
    if i > 1
        plasticity_test[i-1] = mat.variables.cumeq > 0.0
    end
    @test !iszero(tovoigt(mat.variables.stress)[1]) && !iszero(tovoigt(mat.variables.stress)[end])
    @test isapprox(tovoigt(mat.variables.stress; offdiagscale=2.0)[2:5],zeros(4); atol=1e-5)
end
@test any(x->x==1,plasticity_test)
