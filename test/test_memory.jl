# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE
using Test, Tensors
using Materials
parameters = MemoryParameterState(E = 200.0e3,
                                    nu = 0.3,
                                    R0 = 100.0,
                                    Kn = 20.0,
                                    nn = 3.0,
                                    C1 = 10000.0,
                                    D1 = 100.0,
                                    C2 = 50000.0,
                                    D2 = 1000.0,
                                    Q0 = 100.0,
                                    #QM = 300.0,
                                    QM = 500.0,
                                    mu = 100.0,
                                    b = 30.0,
                                    #eta = 0.0,
                                    # eta = 0.05,
                                    eta = 0.5,
                                    m = 0.5,
                                    pt = 0.0,
                                    xi = 0.3)
                                    # xi = 0.0)
mat = Memory(parameters=parameters)

times = [copy(mat.drivers.time)]
stresses = [copy(tovoigt(mat.variables.stress))]
strains = [copy(tovoigt(mat.drivers.strain; offdiagscale=2.0))]
plastic_strains = [copy(tovoigt(mat.variables.plastic_strain; offdiagscale=2.0))]
cumeqs = [copy(mat.variables.cumeq)]
qs = [copy(mat.variables.q)]
Rs = [copy(mat.variables.R)]
zetas = [copy(tovoigt(mat.variables.zeta; offdiagscale=2.0))]

n_cycles = 30
ppc = 40
t = range(0, n_cycles; length=n_cycles*ppc+1)
dtime = t[end]/(length(t)-1)

# Amplitude 1
ea = 0.003
strains11 = ea*sin.(2*pi*t)
for dstrain11 in diff(strains11)
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
R1 = copy(Rs[end])

# Amplitude 2
ea = 0.005
strains11 = ea*sin.(2*pi*t)
for dstrain11 in diff(strains11)
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
R2 = copy(Rs[end])

# Amplitude 3
ea = 0.007
strains11 = ea*sin.(2*pi*t)
for dstrain11 in diff(strains11)
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
R3 = copy(Rs[end])

# Amplitude 4 - Test evanescence
ea = 0.003
strains11 = ea*sin.(2*pi*t)
for dstrain11 in diff(strains11)
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

for dstrain11 in diff(strains11)
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

for dstrain11 in diff(strains11)
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
R4 = copy(Rs[end])
@test R2>R1
@test R3>R2
@test R4<R3
@test isapprox(R1, R4; atol=1.0)
