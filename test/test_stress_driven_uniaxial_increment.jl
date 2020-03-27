# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE
using Tensors, Materials
dtime = 0.25
# mat = Material(IdealPlastic, tuple())
# mat.properties.youngs_modulus = 200.0e3
# mat.properties.poissons_ratio = 0.3
# mat.properties.yield_stress = 100.0

parameters = DSAParameterState(E = 200.0e3,
                                    nu = 0.3,
                                    R0 = 100.0,
                                    Kn = 100.0,
                                    nn = 10.0,
                                    C1 = 10000.0,
                                    D1 = 100.0,
                                    C2 = 50000.0,
                                    D2 = 1000.0,
                                    Q = 50.0,
                                    b = 0.1,
                                    w = 1.0,
                                    P1 = 100.0,
                                    P2 = 1.0,
                                    m = 2/3,
                                    ba = 0.1,
                                    zeta = 1.0)
mat = DSA(parameters = parameters)

times = [mat.drivers.time]
stresses = [copy(tovoigt(mat.variables.stress))]
strains = [copy(tovoigt(mat.drivers.strain; offdiagscale=2.0))]
dstrain11 = 1e-3*dtime
dtimes = [dtime, dtime, dtime, dtime, 1.0]
dstrains11 = [dstrain11, dstrain11, 0.1*dstrain11, -dstrain11, -4*dstrain11]
for i in 1:length(dtimes)
    dstrain11 = dstrains11[i]
    dtime = dtimes[i]
    uniaxial_increment!(mat, dstrain11, dtime)
    # uniaxial_increment!(mat, dstrain11, dtime; dstrain = copy(tovoigt(mat.ddrivers.strain))*dstrain11/mat.ddrivers.strain[1,1]*dtime/mat.ddrivers.time)
    # mat.time += mat.dtime
    # mat.strain .+= mat.dstrain
    # mat.stress .+= mat.dstress
    update_material!(mat)
    push!(times, mat.drivers.time)
    push!(stresses, copy(tovoigt(mat.variables.stress)))
    push!(strains, copy(tovoigt(mat.drivers.strain; offdiagscale=2.0)))
    #@info(tovoigt(mat.variables.stress), stresses_expected[i])
    #@test isapprox(tovoigt(mat.variables.stress), stresses_expected[i])
    #@test isapprox(tovoigt(mat.drivers.strain; offdiagscale=2.0), strains_expected[i])
end
