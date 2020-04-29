# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Tensors, Materials, Test

parameters = DSAParameterState( E  =   200.0e3,
                                nu =     0.3,
                                R0 =   100.0,
                                Kn =   100.0,
                                nn =    10.0,
                                C1 = 10000.0,
                                D1 =   100.0,
                                C2 = 50000.0,
                                D2 =  1000.0,
                                Q  =    50.0,
                                b  =     0.1,
                                w  =     4e-4,
                                P1 =   100.0,
                                P2 =     4e-2,
                                m  =     0.66,
                                m1 =     6.0,
                                m2 =     6.0,
                                M1 =  6000.0,
                                M2 =  6000.0,
                                ba =     1e4,
                                xi =     1.0)
material = DSA(parameters = parameters)

times    = [material.drivers.time]
stresses = [copy(tovoigt(material.variables.stress))]
strains  = [copy(tovoigt(material.drivers.strain; offdiagscale = 2.0))]
Ras      = [copy(material.variables.Ra)]
tas      = [copy(material.variables.ta)]

dtime       = 0.25
dtimes      = [dtime, dtime, dtime, dtime, dtime, dtime, dtime, dtime, 4 * dtime, dtime, dtime, dtime, dtime, dtime, 3 * dtime]

dstrain11   = 1e-3 * dtime
dstrains11  = [dstrain11, dstrain11, dstrain11, dstrain11, dstrain11, dstrain11, dstrain11, -dstrain11, -4 * dstrain11, -dstrain11, -dstrain11, -dstrain11, -dstrain11, -dstrain11, 3 * dstrain11]

#dstress11   = 50.0
#dstresses11 = [dstress11, dstress11, dstress11, -3*dstress11, 0.0, dstress11, dstress11, dstress11, dstress11]
#dtimes      = [dtime, dtime, dtime, dtime, 3.6e3, dtime, dtime, dtime, dtime]

#dstress11   = 25.0
#
#dstresses11 = [dstress11, dstress11, dstress11, dstress11, dstress11,
#               dstress11, dstress11, dstress11, dstress11, dstress11,
#               -10*dstress11, 0.0, dstress11, dstress11, dstress11,
#               dstress11, dstress11, dstress11, dstress11, dstress11,
#               dstress11, dstress11, dstress11, dstress11, dstress11,
#               dstress11, dstress11]
#
#dtimes      = [dtime, dtime, dtime, dtime, dtime,
#               dtime, dtime, dtime, dtime, dtime,
#               dtime, 3.6e3, dtime, dtime, dtime,
#               dtime, dtime, dtime, dtime, dtime,
#               dtime, dtime, dtime, dtime, dtime,
#               dtime, dtime]

dstress11  = 20.0
dtime      = 0.25
# dstress11  = 5.0
# dtime      = 0.05

dstresses11 = [dstress11, dstress11, dstress11, dstress11, dstress11,
               dstress11, dstress11, dstress11, dstress11, dstress11,
               dstress11, dstress11, dstress11, -13*dstress11, 0.0, #260
               dstress11, dstress11, dstress11, dstress11, dstress11,
               dstress11, dstress11, dstress11, dstress11, dstress11,
               dstress11, dstress11, dstress11, dstress11, dstress11,
               dstress11, 0.5*dstress11] #340
               #320.0, 0.0, 0.0]

dtimes      = [dtime, dtime, dtime, dtime, dtime,
               dtime, dtime, dtime, dtime, dtime,
               dtime, dtime, dtime, dtime, 3.6e3,
               dtime, dtime, dtime, dtime, dtime,
               dtime, dtime, dtime, dtime, dtime,
               dtime, dtime, dtime, dtime, dtime,
               dtime, dtime]
               #dtime, 3.6e3, 0.0]

# Smaller increment test
fdstresses = zeros(1)
fdtimes    = zeros(1)

for i in 1:120
    if i < 51
        push!(fdstresses, dstress11)
        push!(fdtimes, dtime)
    elseif i == 51
        push!(fdstresses, -50*dstress11)
        push!(fdtimes, dtime)
    elseif i == 52
        push!(fdstresses, 0.0)
        push!(fdtimes, 3.6e3)
    else
        push!(fdstresses, dstress11)
        push!(fdtimes, dtime)
    end
end
# dstresses11 = fdstresses
# dtimes      = fdtimes

stresses_expected = [[  0.0,    0.0, 0.0, 0.0, 0.0, 0.0],
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

strains_expected = [[ 0.0,      0.0,      0.0,     0.0, 0.0, 0.0],
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
#######
stresses_expected = [[  0.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [  20.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [  40.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [  60.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [  80.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [ 100.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [ 120.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [ 140.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [ 160.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [ 180.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [ 200.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [ 220.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [ 240.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [   0.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [   0.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [  20.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [  40.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [  60.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [  80.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [ 100.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [ 120.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [ 140.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [ 160.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [ 180.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [ 200.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [ 220.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [ 240.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [ 260.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [ 280.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [ 300.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [ 320.0,    0.0, 0.0, 0.0, 0.0, 0.0],
                     [ 340.0,    0.0, 0.0, 0.0, 0.0, 0.0]]

strains_expected =  [[0.0,      0.0,      0.0,     0.0, 0.0, 0.0],
                     [0.0001,  -3.0e-5,  -3.0e-5,  0.0, 0.0, 0.0],
                     [0.0002,  -6.0e-5,  -6.0e-5,  0.0, 0.0, 0.0],
                     [0.0003,  -9.0e-5,  -9.0e-5,  0.0, 0.0, 0.0],
                     [0.0004,  -0.00012, -0.00012, 0.0, 0.0, 0.0],
                     [0.0005,  -0.00015, -0.00015, 0.0, 0.0, 0.0],
                     [0.0006,  -0.00018, -0.00018, 0.0, 0.0, 0.0],
                     [0.00072, -0.00022, -0.00022, 0.0, 0.0, 0.0],
                     [0.00101, -0.00034, -0.00034, 0.0, 0.0, 0.0],
                     [0.00151, -0.00058, -0.00058, 0.0, 0.0, 0.0],
                     [0.00227, -0.00093, -0.00093, 0.0, 0.0, 0.0],
                     [0.0034,  -0.00148, -0.00148, 0.0, 0.0, 0.0],
                     [0.00517, -0.00235, -0.00235, 0.0, 0.0, 0.0],
                     [0.00397, -0.00199, -0.00199, 0.0, 0.0, 0.0],
                     [0.00397, -0.00199, -0.00199, 0.0, 0.0, 0.0],
                     [0.00407, -0.00202, -0.00202, 0.0, 0.0, 0.0],
                     [0.00417, -0.00205, -0.00205, 0.0, 0.0, 0.0],
                     [0.00427, -0.00208, -0.00208, 0.0, 0.0, 0.0],
                     [0.00437, -0.00211, -0.00211, 0.0, 0.0, 0.0],
                     [0.00447, -0.00214, -0.00214, 0.0, 0.0, 0.0],
                     [0.00457, -0.00217, -0.00217, 0.0, 0.0, 0.0],
                     [0.00467, -0.0022,  -0.0022,  0.0, 0.0, 0.0],
                     [0.00477, -0.00223, -0.00223, 0.0, 0.0, 0.0],
                     [0.00487, -0.00226, -0.00226, 0.0, 0.0, 0.0],
                     [0.00497, -0.00229, -0.00229, 0.0, 0.0, 0.0],
                     [0.00509, -0.00232, -0.00232, 0.0, 0.0, 0.0],
                     [0.00525, -0.00238, -0.00238, 0.0, 0.0, 0.0],
                     [0.00549, -0.00248, -0.00248, 0.0, 0.0, 0.0],
                     [0.00596, -0.0027,  -0.0027,  0.0, 0.0, 0.0],
                     [0.00713, -0.00327, -0.00327, 0.0, 0.0, 0.0],
                     [0.00947, -0.00442, -0.00442, 0.0, 0.0, 0.0],
                     [0.01341, -0.00636, -0.00636, 0.0, 0.0, 0.0]]

material2 = DSA(parameters = parameters)
times2    = [material2.drivers.time]
stresses2 = [copy(tovoigt(material2.variables.stress))]
strains2  = [copy(tovoigt(material2.drivers.strain; offdiagscale = 2.0))]

for i in 1:16
#     dtime = 0.25
#     dstress11 = 20.0
# for i in 1:65
#     dtime = 0.05
#     dstress11 = 5.0
    stress_driven_uniaxial_increment!(material2, dstress11, dtime)
    update_material!(material2)
    push!(times2, material2.drivers.time)
    push!(stresses2, copy(tovoigt(material2.variables.stress)))
    push!(strains2, copy(tovoigt(material2.drivers.strain; offdiagscale = 2.0)))
end

for i in 1:length(dtimes)
    dtime = dtimes[i]
    # dstrain11 = dstrains11[i]
    # uniaxial_increment!(material, dstrain11, dtime)
    
    dstress11 = dstresses11[i]
    stress_driven_uniaxial_increment!(material, dstress11, dtime)

    update_material!(material)
    push!(times, material.drivers.time)
    push!(stresses, copy(tovoigt(material.variables.stress)))
    push!(strains, copy(tovoigt(material.drivers.strain; offdiagscale = 2.0)))
    push!(Ras, material.variables.Ra)
    push!(tas, material.variables.ta)
    #@test isapprox(stresses[i], stresses_expected[i]; atol = 1e-4) # Test(1) 
    #@test isapprox(strains[i], strains_expected[i]; atol = 1e-5)   # Test(2)
end

using PyPlot
x11 = [a[1] for a in strains]
y11 = [a[1] for a in stresses]
x112 = [a[1] for a in strains2]
y112 = [a[1] for a in stresses2]

x11p = [round(a[1], digits=5) for a in strains]
y11p = [round(a[1], digits=4) for a in stresses]

RasNorm = [Ra/maximum(Ras) for Ra in Ras]
tasNorm = [ta/maximum(tas) for ta in tas]

#println("Stresses: ", y11p)
#println("-------")
#println("Strains:  ", x11p)
#println("*_*_*_*")
#for a in strains
#    a[1] = round(a[1], digits=5)
#    a[2] = round(a[2], digits=5)
#    a[3] = round(a[3], digits=5)
#    println(a)
#end

clf()

fig = figure("test_DSA.jl",figsize=(5,9)) # Create a new blank figure

subplot(211)
plot(x11,y11, label="interrupted")
plot(x112,y112,linestyle="--", label="uninterrupted")
title("Vetokoe")
xlabel("Strain, \$\\varepsilon_{11}\$")
ylabel("Stress, \$\\sigma_{11}\$")
legend()

subplot(212)
plot(times, RasNorm, label="\$R_a\$")
plot(times, tasNorm, linestyle="--", label="\$t_a\$")
title("Normalized Evolution of \$R_a\$ & \$t_a\$")
xlabel("Time")
ylabel("Ra, ta")
legend()

fig.canvas.draw() # Update the figure
# suptitle("test_DSA.jl")
show()
savefig("test_DSA.jl.png")
gcf()