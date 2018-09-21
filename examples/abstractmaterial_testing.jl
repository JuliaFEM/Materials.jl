mutable struct Variable{T}
    value :: T
    dvalue :: T
end

function Variable(x)
    return Variable(x, zero(x))
end

reset!(v::Variable) = (v.dvalue = zero(v.value))
update!(v::Variable) = (v.value += v.dvalue; reset!(v))
update!(v::Variable{<:Array}) = v.value .= v.value .+ v.dvalue

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

println("Benchmark Variable{Float64}")
@btime bench_float64()
println("Benchmark Variable{Array{Float64,1}}")
@btime bench_array()
println("Benchmark Variable{Tensor{2,3,Float64,9}}")
@btime bench_tensor()
