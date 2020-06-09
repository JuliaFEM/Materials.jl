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

parameters = ChabocheParameterState(E=200.0e3,
                                    nu=0.3,
                                    R0=100.0,
                                    Kn=100.0,
                                    nn=10.0,
                                    C1=10000.0,
                                    D1=100.0,
                                    C2=50000.0,
                                    D2=1000.0,
                                    Q=50.0,
                                    b=0.1)
chabmat = Chaboche(parameters = parameters)
s33s = [chabmat.variables.stress[3,3]]
for i=2:length(ts)
    dtime = ts[i] - ts[i-1]
    dstrain = fromvoigt(Symm2{Float64}, strains[i] - strains[i-1]; offdiagscale=2.0)
    chabmat.ddrivers = ChabocheDriverState(time = dtime, strain = dstrain)
    integrate_material!(chabmat)
    update_material!(chabmat)
    push!(s33s, chabmat.variables.stress[3,3])
end
@test isapprox(s33s, s33_; rtol=0.05)
