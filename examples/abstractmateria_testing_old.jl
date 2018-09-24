using Tensors, BenchmarkTools

abstract type AbstractMaterialStateOld end

mutable struct VariableStateOld <: AbstractMaterialStateOld
     stress::Vector{Float64}
    dstress::Vector{Float64}
    strain::Vector{Float64}
    dstrain::Vector{Float64}
    backstress1::Vector{Float64}
    dbackstress1::Vector{Float64}
    backstress2::Vector{Float64}
    dbackstress2::Vector{Float64}
    plastic_strain::Vector{Float64}
    dplastic_strain::Vector{Float64}
    cumeq::Float64
    dcumeq::Float64
    R::Float64
    dR::Float64
end

function update!(state::VariableStateOld)
    state.stress[:] .+= state.dstress[:]
    state.strain[:] .+= state.dstrain[:]
    state.backstress1[:] .+= state.dbackstress1[:]
    state.backstress2[:] .+= state.dbackstress2[:]
    state.plastic_strain[:] .+= state.dplastic_strain[:]
    state.cumeq += state.dcumeq
    state.R += state.dR
    fill!(state.dstress, 0.0)
    fill!(state.dstrain, 0.0)
    fill!(state.dbackstress1, 0.0)
    fill!(state.dbackstress2, 0.0)
    fill!(state.dplastic_strain, 0.0)
    state.dcumeq = 0.0
    state.dR = 0.0
end

function bench_chaboche_style_oldstate(N)
    stress = zeros(6)
    dstress = zeros(6)
    strain = zeros(6)
    dstrain = zeros(6)
    backstress1 = zeros(6)
    dbackstress1 = zeros(6)
    backstress2 = zeros(6)
    dbackstress2 = zeros(6)
    plastic_strain = zeros(6)
    dplastic_strain = zeros(6)
    cumeq = 0.0
    dcumeq = 0.0
    R = 0.0
    dR = 0.0
    state = VariableStateOld(stress, dstress, strain, dstrain,
        backstress1, dbackstress1, backstress2, dbackstress2,
        plastic_strain, dplastic_strain, cumeq, dcumeq, R, dR)

    for i in 1:N
        state.dstress[:] .= randn(6)
        state.dstrain[:] .= randn(6)
        state.dbackstress1[:] .= randn(6)
        state.dbackstress2[:] .= randn(6)
        state.dplastic_strain[:] .= randn(6)
        state.dcumeq = norm(dplastic_strain)
        state.dR = randn()
        update!(state) # Update paremeters
    end
end
N = 1000
println("Benchmark Chaboche Old State")
@btime bench_chaboche_style_oldstate(N)
