mutable struct Variable{T}
    value :: T
    dvalue :: T
end

function Variable(x)
    return Variable(x, zero(x))
end

reset!(v::Variable) = (v.dvalue = zero(v.value))
update!(v::Variable) = (v.value += v.dvalue; reset!(v))
update!(v::Variable{<:Array}) = v.value .+= v.dvalue

using Tensors
a = 1.0
b = [1.0, 2.0, 3.0]
c = Tensor{2, 3}([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
vara = Variable(a)
varb = Variable(b)
varc = Variable(c)

@info "Initial state: $vara"
vara.dvalue += rand()
@info "After setting dvalue: $vara"
update!(vara)
@info "After update!: $vara"

@info "Initial state: $varb"
varb.dvalue += rand(3)
@info "After setting dvalue: $varb"
update!(varb)
@info "After update!: $varb"

@info "Initial state: $varc"
varc.dvalue += Tensor{2,3}(rand(9))
@info "After setting dvalue: $varc"
update!(varc)
@info "After update!: $varc"

using BenchmarkTools
N = 1000
function bench_float64()
    # Random walk test"
    var = Variable(1.0)
    for i in 1:N
        var.dvalue += randn()
        update!(var)
    end
    return var
end
function bench_array()
    # Random walk test
    var = Variable([1.0, 2.0, 3.0])
    for i in 1:N
        var.dvalue += randn(3)
        update!(var)
    end
    return var
end
function bench_tensor()
    # Random walk test
    var = Variable(Tensor{2, 3}([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]))
    for i in 1:N
        var.dvalue += randn(Tensor{2,3})
        update!(var)
    end
end
function bench_symtensor()
    # Random walk test
    var = Variable(Symm2([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    for i in 1:N
        var.dvalue += randn(Symm2{Float64})
        update!(var)
    end
end

# println("Benchmark Variable{Float64}")
# @btime bench_float64()
# println("Benchmark Variable{Array{Float64,1}}")
# @btime bench_array()
# println("Benchmark Variable{Tensor{2,3,Float64,9}}")
# @btime bench_tensor()
# println("Benchmark Variable{SymmetricTensor{2,3,Float64,6}}")
# @btime bench_symtensor()

abstract type AbstractVariableState end

mutable struct TestState <: AbstractVariableState
    x :: Variable{Float64}
end

mutable struct VariableState <: AbstractVariableState
   stress::Variable{SymmetricTensor{2,3,Float64,6}}
   strain::Variable{SymmetricTensor{2,3,Float64,6}}
   backstress1::Variable{SymmetricTensor{2,3,Float64,6}}
   backstress2::Variable{SymmetricTensor{2,3,Float64,6}}
   plastic_strain::Variable{SymmetricTensor{2,3,Float64,6}}
   cumeq::Variable{Float64}
   R::Variable{Float64}
end

function update!(state::T) where {T<:AbstractVariableState}
    for fn in fieldnames(T)
        update!(getfield(state, fn))
    end
end

function bench_chaboche_style_variablestate()
    stress = Variable(zero(Symm2))
    strain = Variable(zero(Symm2))
    backstress1 = Variable(zero(Symm2))
    backstress2 = Variable(zero(Symm2))
    plastic_strain = Variable(zero(Symm2))
    cumeq = Variable(0.0)
    R = Variable(0.0)
    state = VariableState(stress, strain, backstress1,
                        backstress2, plastic_strain, cumeq, R)
    for i in 1:N
        state.stress.dvalue = randn(Symm2)
        state.strain.dvalue = randn(Symm2)
        state.backstress1.dvalue = randn(Symm2)
        state.backstress2.dvalue = randn(Symm2)
        state.plastic_strain.dvalue = randn(Symm2)
        state.cumeq.dvalue = norm(state.plastic_strain.dvalue)
        state.R.dvalue = randn()
        update!(state)
    end
    return state
end

println("Benchmark Chaboche VariableState")
@btime bench_chaboche_style_variablestate()
