# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using FEMMaterials, Test
using DelimitedFiles


path = joinpath("test_chaboche", "unitelement_results.rpt")
data = readdlm(path, Float64; skipstart=4)
ts = data[:,1]
s11_ = data[:,2]
s12_ = data[:,3]
s13_ = data[:,4]
s22_ = data[:,5]
s23_ = data[:,6]
s33_ = data[:,7]
e11_ = data[:,8]
e12_ = data[:,9]
e13_ = data[:,10]
e22_ = data[:,11]
e23_ = data[:,12]
e33_ = data[:,13]

strains = [[e11_[i], e22_[i], e33_[i], e23_[i], e13_[i], e12_[i]] for i in 1:length(ts)]

mat = Material(Chaboche, ())
mat.properties.youngs_modulus = 200.0e3
mat.properties.poissons_ratio = 0.3
mat.properties.yield_stress = 100.0
mat.properties.K_n = 100.0
mat.properties.n_n = 10.0
mat.properties.C_1 = 10000.0
mat.properties.D_1 = 100.0
mat.properties.C_2 = 50000.0
mat.properties.D_2 = 1000.0
mat.properties.Q = 50.0
mat.properties.b = 0.1

mat.stress = zeros(6)
sim = Simulator(mat)
FEMMaterials.initialize!(sim, strains, ts)
t0 = time()
FEMMaterials.run!(sim)
t1 = time()
@info "Test took $(t1-t0)s!"
s33s = [s[3] for s in sim.stresses]
@test isapprox(s33s, s33_; rtol=0.001)
