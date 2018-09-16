using Materials, Test
using Materials: Simulator
using PyPlot

num = 100 # Number of steps
E = 200000.0 # Young's modulus [GPa]
poissons_ratio = 0.3 # Poisson's ratio
stress_y = 200.0 # Yield stress
dt = 0.1
n = 0.92
K = 180.0e3


# Simulation 1
mat = Material(ViscoPlastic, (:norton, [K, n]))
mat.properties.youngs_modulus = E
mat.properties.poissons_ratio = poissons_ratio
mat.properties.yield_stress = stress_y
times = [0.0]
strains = [zeros(6)]

L = 1000.0               # Initial length [mm]
dL = range(0, stop=3.1, length=num) # Change of length [mm]

strain = collect(dL / L)
# strain_total = vec([strain; reverse(strain[2:end-1]); -strain; reverse(-strain[2:end-1])])
strain_total = strain
nsteps = length(strain_total)
dt = 0.03

for i=1:nsteps
    str = zeros(6)
    str[1] = strain_total[i]
    str[2] = -strain_total[i] * poissons_ratio
    str[3] = -strain_total[i] * poissons_ratio

    push!(times, times[end] + dt)
    push!(strains, str)
end

println(strains[2][1] / dt)
sim = Simulator(mat)
Materials.initialize!(sim, strains, times)
Materials.run!(sim)

s11s = [s[1] for s in sim.stresses]
e11s = [e[1] for e in strains]

# Simulation 2
mat = Material(ViscoPlastic, (:norton, [K, n]))
mat.properties.youngs_modulus = E
mat.properties.poissons_ratio = poissons_ratio
mat.properties.yield_stress = stress_y
times = [0.0]
strains = [zeros(6)]

dt = 0.08

for i=1:nsteps
    str = zeros(6)
    str[1] = strain_total[i]
    str[2] = -strain_total[i] * poissons_ratio
    str[3] = -strain_total[i] * poissons_ratio

    push!(times, times[end] + dt)
    push!(strains, str)
end
println(strains[3][1] / dt)
sim = Simulator(mat)
Materials.initialize!(sim, strains, times)
Materials.run!(sim)

s11s2 = [s[1] for s in sim.stresses]
e11s2 = [e[1] for e in strains]

# Simulation 3
mat = Material(ViscoPlastic, (:norton, [K, n]))
mat.properties.youngs_modulus = E
mat.properties.poissons_ratio = poissons_ratio
mat.properties.yield_stress = stress_y
times = [0.0]
strains = [zeros(6)]

dt = 0.5

for i=1:nsteps
    str = zeros(6)
    str[1] = strain_total[i]
    str[2] = -strain_total[i] * poissons_ratio
    str[3] = -strain_total[i] * poissons_ratio

    push!(times, times[end] + dt)
    push!(strains, str)
end
println(strains[3][1] / dt)
sim = Simulator(mat)
Materials.initialize!(sim, strains, times)
Materials.run!(sim)

s11s3 = [s[1] for s in sim.stresses]
e11s3 = [e[1] for e in strains]
legend(bbox_to_anchor=[1.05,1],loc=2,borderaxespad=0)
plot(e11s, s11s, label="de/dt=0.001346")
plot(e11s2, s11s2, label="de/dt=0.00081")
plot(e11s3, s11s3, label="de/dt=0.000008")
grid()
show()