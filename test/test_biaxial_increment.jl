# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE
using Test, Tensors
dtime = 0.25
parameters = IdealPlasticParameterState(youngs_modulus = 200.0e3,
                                        poissons_ratio = 0.3,
                                        yield_stress = 100.0)
mat = IdealPlastic(parameters=parameters)
times = [mat.drivers.time]
stresses = [copy(tovoigt(mat.variables.stress))]
dstrain11 = 1e-3*dtime
dstrain12 = 1e-3*dtime

dtimes = [dtime, dtime, dtime, dtime, 1.0]
dstrains11 = [dstrain11, dstrain11, dstrain11, -dstrain11, -4*dstrain11]
dstrains12 = [dstrain12, dstrain12, dstrain12, -dstrain12, -4*dstrain12]
plasticity_test = zeros(length(dstrains11)-1)
multiaxial_test = zeros(length(dstrains11))
for i in 1:length(dtimes)
    dstrain11 = dstrains11[i]
    dtime = dtimes[i]
    biaxial_increment!(mat, dstrain11, dstrain12, dtime)
    update_material!(mat)
    push!(stresses, copy(tovoigt(mat.variables.stress)))
    if i > 1
        # Check if test reaches plasticity by looking if stresses remain linear
        plasticity_test[i-1] = !(isapprox(stresses[i+1][1], stresses[i][1]
         + dstrains11[i]/dstrains11[1]*stresses[2][1]; atol=1e-9))
    end
    if !(all(x->isapprox(x,0.0;atol = atol=1e-8),tovoigt(mat.variables.stress; offdiagscale=2.0)[2:5]))
        # Check if test has multiaxial stress
        multiaxial_test[i] = 1
    end
end
@test any(x->x==1,plasticity_test)
@test any(x->x==1,multiaxial_test)
