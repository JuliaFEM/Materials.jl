# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE
using Materials, Test
dtime = 0.25
mat = Material(IdealPlastic, tuple())
mat.properties.youngs_modulus = 200.0e3
mat.properties.poissons_ratio = 0.3
mat.properties.yield_stress = 100.0
times = [mat.time]
stresses = [copy(mat.stress)]
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
dstrains11 = [dstrain11, dstrain11, dstrain11, -dstrain11, -dstrain11]
for i in 1:length(dtimes)
    dstrain11 = dstrains11[i]
    dtime = dtimes[i]
    if i == 1
        uniaxial_increment!(mat, dstrain11, dtime)
    else
        uniaxial_increment!(mat, dstrain11, dtime; dstrain = copy(mat.dstrain)*dstrain11/mat.dstrain[1]*dtime/mat.dtime)
    end
    mat.time += mat.dtime
    mat.strain .+= mat.dstrain
    mat.stress .+= mat.dstress
    push!(times, mat.time)
    push!(stresses, copy(mat.stress))
    @test isapprox(mat.stress, stresses_expected[i])
    @test isapprox(mat.strain, strains_expected[i])
end
