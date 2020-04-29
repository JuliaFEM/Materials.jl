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
                                w  =     1e-5,
                                P1 =   200.0,
                                P2 =     1e-1,
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

dstrain11   = 2e-4 * dtime # Corresponds to 10 MPa elastic stress response

material2 = DSA(parameters = parameters)
times2    = [material2.drivers.time]
stresses2 = [copy(tovoigt(material2.variables.stress))]
strains2  = [copy(tovoigt(material2.drivers.strain; offdiagscale = 2.0))]

no_steps = 100
# Uninterrupted test
for i in 1:no_steps
    uniaxial_increment!(material2, dstrain11, dtime)
    update_material!(material2)
    push!(times2, material2.drivers.time)
    push!(stresses2, copy(tovoigt(material2.variables.stress)))
    push!(strains2, copy(tovoigt(material2.drivers.strain; offdiagscale = 2.0)))
end

# Interrupted test
n_interrupt = 40
for i in 1:n_interrupt
    uniaxial_increment!(material, dstrain11, dtime)
    update_material!(material)
    push!(times, material.drivers.time)
    push!(stresses, copy(tovoigt(material.variables.stress)))
    push!(strains, copy(tovoigt(material.drivers.strain; offdiagscale = 2.0)))
    push!(Ras, material.variables.Ra)
    push!(tas, material.variables.ta)
end

# Interruption and hold
# Drive to zero stress
strain_at_stop = material.drivers.strain[1,1]
stress_driven_uniaxial_increment!(material, -material.variables.stress[1,1], dtime)
update_material!(material)
push!(times, material.drivers.time)
push!(stresses, copy(tovoigt(material.variables.stress)))
push!(strains, copy(tovoigt(material.drivers.strain; offdiagscale = 2.0)))
push!(Ras, material.variables.Ra)
push!(tas, material.variables.ta)
# Hold for 3600 seconds
stress_driven_uniaxial_increment!(material, 0.0, 3600)
update_material!(material)
push!(times, material.drivers.time)
push!(stresses, copy(tovoigt(material.variables.stress)))
push!(strains, copy(tovoigt(material.drivers.strain; offdiagscale = 2.0)))
push!(Ras, material.variables.Ra)
push!(tas, material.variables.ta)
# Continue test
dstrain_extra = strain_at_stop - material.drivers.strain[1,1]
no_extra_steps = Int(ceil(dstrain_extra/dstrain11))
for i in n_interrupt+1:no_steps+no_extra_steps
    uniaxial_increment!(material, dstrain11, dtime)
    update_material!(material)
    push!(times, material.drivers.time)
    push!(stresses, copy(tovoigt(material.variables.stress)))
    push!(strains, copy(tovoigt(material.drivers.strain; offdiagscale = 2.0)))
    push!(Ras, material.variables.Ra)
    push!(tas, material.variables.ta)
end


using PyPlot
x11 = [a[1] for a in strains]
y11 = [a[1] for a in stresses]
x112 = [a[1] for a in strains2]
y112 = [a[1] for a in stresses2]

x11p = [round(a[1], digits=5) for a in strains]
y11p = [round(a[1], digits=4) for a in stresses]

RasNorm = [Ra/parameters.P1 for Ra in Ras]
tasNorm = [ta/maximum(tas) for ta in tas]

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
xlim([3600.0, maximum(times)])
title("Normalized Evolution of \$R_a\$ & \$t_a\$")
xlabel("Time")
ylabel("Ra, ta")
legend()

fig.canvas.draw() # Update the figure
# suptitle("test_DSA.jl")
show()
savefig("test_DSA.jl.png")
gcf()
