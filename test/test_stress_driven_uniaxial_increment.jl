# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE
using Test, Tensors, Materials, Plots
dtime = 0.25
# material = Material(IdealPlastic, tuple())
# material.properties.youngs_modulus = 200.0e3
# material.properties.poissons_ratio = 0.3
# material.properties.yield_stress = 100.0
parameters = IdealPlasticParameterState(youngs_modulus = 200.0e3,
                                        poissons_ratio = 0.3,
                                        yield_stress = 100.0)
material = IdealPlastic(parameters=parameters)
times = [material.drivers.time]
stresses = [copy(tovoigt(material.variables.stress))]
stresses_expected = [[       50.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [      100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [      100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [      100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [      100.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
dstrain11 = 1e-3*dtime # == 1/200e3 * 50.0 

E       = material.parameters.youngs_modulus
nu      = material.parameters.poissons_ratio

strains_expected = [[(1/E)*stresses_expected[1][1], -(nu/E)*stresses_expected[1][1], -(nu/E)*stresses_expected[1][1], 0.0, 0.0, 0.0],
                    [(1/E)*stresses_expected[2][1], -(nu/E)*stresses_expected[2][1], -(nu/E)*stresses_expected[2][1], 0.0, 0.0, 0.0],
                    [(1/E)*stresses_expected[3][1], -(nu/E)*stresses_expected[3][1], -(nu/E)*stresses_expected[3][1], 0.0, 0.0, 0.0].*2,
                    [(1/E)*stresses_expected[4][1], -(nu/E)*stresses_expected[4][1], -(nu/E)*stresses_expected[4][1], 0.0, 0.0, 0.0].*3,
                    [(1/E)*stresses_expected[5][1], -(nu/E)*stresses_expected[5][1], -(nu/E)*stresses_expected[5][1], 0.0, 0.0, 0.0].*4]
dtimes      = [dtime, dtime, dtime, dtime, 1.0]
dstrains11  = [dstrain11, dstrain11, dstrain11, dstrain11, dstrain11]
for i in 1:length(dtimes)
    dstrain11 = dstrains11[i]
    dtime = dtimes[i]
    stress_driven_uniaxial_increment!(material, dstrain11, dtime)
    # stress_driven_uniaxial_increment!(material, dstrain11, dtime; dstrain = copy(tovoigt(material.ddrivers.strain))*dstrain11/material.ddrivers.strain[1,1]*dtime/material.ddrivers.time)
    # material.time += material.dtime
    # material.strain .+= material.dstrain
    # material.stress .+= material.dstress
    update_material!(material)
    # push!(times, material.drivers.time)
    # push!(stresses, copy(tovoigt(material.variables.stress)))
    #@info(tovoigt(material.variables.stress), stresses_expected[i])
    #@test isapprox(tovoigt(material.variables.stress), stresses_expected[i])
    #@test isapprox(tovoigt(material.drivers.strain; offdiagscale=2.0), strains_expected[i])
end

xx = [0,strains_expected[1][1],strains_expected[2][1],strains_expected[3][1],strains_expected[4][1],strains_expected[5][1]]
yy = [0,stresses_expected[1][1],stresses_expected[2][1],stresses_expected[3][1],stresses_expected[4][1],stresses_expected[5][1]]

pyplot()
blot=plot(xx, yy,
            title = "Stress-driven Uniaxial Increment, Ideal Plastic (11)",
            xlabel = "Stress",
            ylabel = "Strain",
            xlims = (0,0.002),
            ylims = (0,125),
            yticks = 0:25:125)
display(blot)
png(blot,"SDUI.png")