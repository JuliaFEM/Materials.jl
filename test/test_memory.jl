# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE
using Test, Tensors

parameters = MemoryParameterState(E = 200.0e3,
                                    nu = 0.3,
                                    R0 = 100.0,
                                    Kn = 100.0,
                                    nn = 10.0,
                                    C1 = 10000.0,
                                    D1 = 100.0,
                                    C2 = 50000.0,
                                    D2 = 1000.0,
                                    Q0 = 0.0,
                                    #QM = 300.0,
                                    QM = 0.0,
                                    mu = 0.1,
                                    b = 0.1,
                                    #eta = 0.0,
                                    # eta = 0.05,
                                    eta = 0.5,
                                    m = 3.0,
                                    pt = 10.0,
                                    xi = 0.0)
                                    # xi = 0.0)
mat = Memory(parameters=parameters)
# parameters = ChabocheParameterState(E = 200.0e3,
#                                     nu = 0.3,
#                                     R0 = 100.0,
#                                     Kn = 100.0,
#                                     nn = 10.0,
#                                     C1 = 10000.0,
#                                     D1 = 100.0,
#                                     C2 = 50000.0,
#                                     D2 = 1000.0,
#                                     Q = 50.0,
#                                     b = 0.1)
# mat = Chaboche(parameters = parameters)

times = [copy(mat.drivers.time)]
stresses = [copy(tovoigt(mat.variables.stress))]
strains = [copy(tovoigt(mat.drivers.strain; offdiagscale=2.0))]
plastic_strains = [copy(tovoigt(mat.variables.plastic_strain; offdiagscale=2.0))]
cumeqs = [copy(mat.variables.cumeq)]
qs = [copy(mat.variables.q)]
Rs = [copy(mat.variables.q)]
zetas = [copy(tovoigt(mat.variables.zeta; offdiagscale=2.0))]

n_cycles = 10
ppc = 40
t = range(0, n_cycles; length=n_cycles*ppc+1)

# Amplitude 1
ea = 0.003
strains11 = ea*sin.(2*pi*t)
for (dtime, dstrain11) in zip(diff(t), diff(strains11))
    # println("dtime: ", dtime, " dstrain: ", dstrain11)
    uniaxial_increment!(mat, dstrain11, dtime)
    update_material!(mat)
    push!(times, mat.drivers.time)
    push!(stresses, copy(tovoigt(mat.variables.stress)))
    push!(strains, copy(tovoigt(mat.drivers.strain; offdiagscale=2.0)))
    push!(plastic_strains, copy(tovoigt(mat.variables.plastic_strain; offdiagscale=2.0)))
    push!(cumeqs, copy(mat.variables.cumeq))
    push!(qs, copy(mat.variables.q))
    push!(Rs, copy(mat.variables.R))
    push!(zetas, copy(tovoigt(mat.variables.zeta; offdiagscale=2.0)))
end
