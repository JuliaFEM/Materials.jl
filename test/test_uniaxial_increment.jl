# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE
using Test, Tensors
dtime = 0.25
# mat = Material(IdealPlastic, tuple())
# mat.properties.youngs_modulus = 200.0e3
# mat.properties.poissons_ratio = 0.3
# mat.properties.yield_stress = 100.0
parameters = IdealPlasticParameterState(youngs_modulus = 200.0e3,
                                        poissons_ratio = 0.3,
                                        yield_stress = 100.0)
mat = IdealPlastic(parameters=parameters)
times = [mat.drivers.time]
stresses = [copy(tovoigt(mat.variables.stress))]
stresses_expected = [[50.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [50.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [-100.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
dstrain11 = 1e-3*dtime
strains_expected = [[dstrain11, -0.3*dstrain11, -0.3*dstrain11, 0.0, 0.0, 0.0],
                    [2*dstrain11, -0.3*dstrain11*2, -0.3*dstrain11*2, 0.0, 0.0, 0.0],
                    [3*dstrain11, -0.3*dstrain11*2 - 0.5*dstrain11, -0.3*dstrain11*2 - 0.5*dstrain11, 0.0, 0.0, 0.0],
                    [2*dstrain11, -0.3*dstrain11 - 0.5*dstrain11, -0.3*dstrain11 - 0.5*dstrain11, 0.0, 0.0, 0.0],
                    [-2*dstrain11, 0.3*dstrain11*2, 0.3*dstrain11*2, 0.0, 0.0, 0.0]]
dtimes = [dtime, dtime, dtime, dtime, 1.0]
dstrains11 = [dstrain11, dstrain11, dstrain11, -dstrain11, -4*dstrain11]
for i in 1:length(dtimes)
    dstrain11 = dstrains11[i]
    dtime = dtimes[i]
    uniaxial_increment!(mat, dstrain11, dtime)
    # uniaxial_increment!(mat, dstrain11, dtime; dstrain = copy(tovoigt(mat.ddrivers.strain))*dstrain11/mat.ddrivers.strain[1,1]*dtime/mat.ddrivers.time)
    # mat.time += mat.dtime
    # mat.strain .+= mat.dstrain
    # mat.stress .+= mat.dstress
    update_material!(mat)
    # push!(times, mat.drivers.time)
    # push!(stresses, copy(tovoigt(mat.variables.stress)))
    #@info(tovoigt(mat.variables.stress), stresses_expected[i])
    @test isapprox(tovoigt(mat.variables.stress), stresses_expected[i])
    @test isapprox(tovoigt(mat.drivers.strain; offdiagscale=2.0), strains_expected[i])
end
