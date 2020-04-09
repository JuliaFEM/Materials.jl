# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Tensors, Materials, Test

parameters = DSAParameterState( E    =   200.0e3,
                                nu   =     0.3,
                                R0   =   100.0,
                                Kn   =   100.0,
                                nn   =    10.0,
                                C1   = 10000.0,
                                D1   =   100.0,
                                C2   = 50000.0,
                                D2   =  1000.0,
                                Q    =    50.0,
                                b    =     0.1,
                                w    =     1.0,
                                P1   =   100.0,
                                P2   =     1.0,
                                m    =     0.66,
                                m1   =     6.0,
                                m2   =     6.0,
                                M1   =  6000.0,
                                M2   =  6000.0,
                                ba   =     0.1,
                                zeta =     1.0)
material = DSA(parameters = parameters)

times    = [material.drivers.time]
stresses = [copy(tovoigt(material.variables.stress))]
strains  = [copy(tovoigt(material.drivers.strain; offdiagscale = 2.0))]

dtime = 0.25
dtimes      = [dtime, dtime, dtime, dtime, dtime, dtime, dtime, dtime, 4 * dtime, dtime, dtime, dtime, dtime, dtime, 3 * dtime]

dstrain11   = 1e-3 * dtime
dstrains11  = [dstrain11, dstrain11, dstrain11, dstrain11, dstrain11, dstrain11, dstrain11, -dstrain11, -4 * dstrain11, -dstrain11, -dstrain11, -dstrain11, -dstrain11, -dstrain11, 3 * dstrain11]

stresses_expected = [[   0.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                    [  50.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                    [ 100.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                    [ 143.1135, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [ 158.9724, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [ 168.6481, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [ 176.7579, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [ 183.8679, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [ 133.8679, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [ -66.1321, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-109.3857, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-127.9448, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-140.7376, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-151.7599, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-161.5031, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [ -11.5031, 0.0, 0.0, 0.0, 0.0, 0.0]]

strains_expected  = [[ 0.0,      0.0,      0.0,     0.0, 0.0, 0.0],
                    [ 0.00025, -7.5e-5,  -7.5e-5,  0.0, 0.0, 0.0],
                    [ 0.0005,  -0.00015, -0.00015, 0.0, 0.0, 0.0],
                    [ 0.00075, -0.00023, -0.00023, 0.0, 0.0, 0.0],
                    [ 0.001,   -0.00034, -0.00034, 0.0, 0.0, 0.0],
                    [ 0.00125, -0.00046, -0.00046, 0.0, 0.0, 0.0],
                    [ 0.0015,  -0.00057, -0.00057, 0.0, 0.0, 0.0],
                    [ 0.00175, -0.00069, -0.00069, 0.0, 0.0, 0.0],
                    [ 0.0015,  -0.00062, -0.00062, 0.0, 0.0, 0.0],
                    [ 0.0005,  -0.00032, -0.00032, 0.0, 0.0, 0.0],
                    [ 0.00025, -0.00023, -0.00023, 0.0, 0.0, 0.0],
                    [ 0.0,     -0.00013, -0.00013, 0.0, 0.0, 0.0],
                    [-0.00025, -1.6e-5,  -1.6e-5,  0.0, 0.0, 0.0],
                    [-0.0005,   9.8e-5,   9.8e-5,  0.0, 0.0, 0.0],
                    [-0.00075,  0.00021,  0.00021, 0.0, 0.0, 0.0],
                    [ 0.0,     -1.2e-5,  -1.2e-5,  0.0, 0.0, 0.0]]

for i in 1:length(dtimes)
    dtime = dtimes[i]
    dstrain11 = dstrains11[i]
    uniaxial_increment!(material, dstrain11, dtime)
    update_material!(material)
    push!(times, material.drivers.time)
    push!(stresses, copy(tovoigt(material.variables.stress)))
    push!(strains, copy(tovoigt(material.drivers.strain; offdiagscale = 2.0)))
    @test isapprox(stresses[i], stresses_expected[i]; atol = 1e-4) # Test(1) 
    @test isapprox(strains[i], strains_expected[i]; atol = 1e-5)   # Test(2)
end