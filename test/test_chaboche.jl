# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Test, Tensors
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

parameters = ChabocheParameterState(E = 200.0e3,
                                    nu = 0.3,
                                    R0 = 100.0,
                                    Kn = 100.0,
                                    nn = 10.0,
                                    C1 = 10000.0,
                                    D1 = 100.0,
                                    C2 = 50000.0,
                                    D2 = 1000.0,
                                    Q = 50.0,
                                    b = 0.1)
chabmat = Chaboche(parameters = parameters)
s33s = [chabmat.variables.stress[3,3]]
for i=2:length(ts)
    dtime = ts[i]-ts[i-1]
    dstrain = fromvoigt(SymmetricTensor{2,3,Float64}, strains[i]-strains[i-1]; offdiagscale=2.0)
    chabmat.ddrivers = ChabocheDriverState(time = dtime, strain = dstrain)
    integrate_material!(chabmat)
    update_material!(chabmat)
    push!(s33s, chabmat.variables.stress[3,3])
end
@test isapprox(s33s, s33_; rtol=0.05)

# function test_chab()
#     mat = Material(Chaboche, ())
#     mat.properties.youngs_modulus = 200.0e3
#     mat.properties.poissons_ratio = 0.3
#     mat.properties.yield_stress = 100.0
#     mat.properties.K_n = 100.0
#     mat.properties.n_n = 10.0
#     mat.properties.C_1 = 10000.0
#     mat.properties.D_1 = 100.0
#     mat.properties.C_2 = 50000.0
#     mat.properties.D_2 = 1000.0
#     mat.properties.Q = 50.0
#     mat.properties.b = 0.1
#
#     mat.stress = zeros(6)
#     s33s = [0.0]
#     for i=2:length(ts)
#         dtime = ts[i]-ts[i-1]
#         dstrain = strains[i]-strains[i-1]
#         mat.dtime = dtime
#         mat.dstrain = dstrain
#         integrate_material!(mat)
#         mat.time += mat.dtime
#         mat.strain .+= mat.dstrain
#         mat.stress .+= mat.dstress
#         mat.properties.plastic_strain .+= mat.properties.dplastic_strain
#         mat.properties.backstress1 .+= mat.properties.dbackstress1
#         mat.properties.backstress2 .+= mat.properties.dbackstress2
#         mat.properties.R += mat.properties.dR
#         mat.properties.cumulative_equivalent_plastic_strain += mat.properties.dcumulative_equivalent_plastic_strain
#         push!(s33s, copy(mat.stress[3]))
#     end
#     # @test isapprox(s33s, s33_; rtol=0.001)
# end

# using BenchmarkTools
# @btime test_chab()
