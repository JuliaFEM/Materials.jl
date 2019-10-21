using Tensors
using BenchmarkTools

abstract type AbstractMaterialState end

@generated function Base.:+(state::T, dstate::T) where {T <: AbstractMaterialState}
   expr = [:(state.$p+ dstate.$p) for p in fieldnames(T)]
   return :(T($(expr...)))
end

struct SomeState <: AbstractMaterialState
   stress::SymmetricTensor{2,3,Float64}
end


state = SomeState(SymmetricTensor{2,3,Float64}([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

N = 1000
function bench_state(N)
   state = SomeState(SymmetricTensor{2,3,Float64}([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
   for i in 1:N
      dstate = SomeState(randn(SymmetricTensor{2,3,Float64}))
      state = state + dstate
   end
   return state
end

# println("Benchmark State{SymmetricTensor{2,3,Float64}}")
# @btime bench_state(N)

struct AnotherState <: AbstractMaterialState
   stress::SymmetricTensor{2,3,Float64,6}
   strain::SymmetricTensor{2,3,Float64,6}
   backstress1::SymmetricTensor{2,3,Float64,6}
   backstress2::SymmetricTensor{2,3,Float64,6}
   plastic_strain::SymmetricTensor{2,3,Float64,6}
   cumeq::Float64
   R::Float64
end
 function bench_chaboche_style_state(N)
   stress = zero(SymmetricTensor{2,3})
   strain = zero(SymmetricTensor{2,3})
   backstress1 = zero(SymmetricTensor{2,3})
   backstress2 = zero(SymmetricTensor{2,3})
   plastic_strain = zero(SymmetricTensor{2,3})
   cumeq = 0.0
   R = 0.0
   state = AnotherState(stress, strain, backstress1,
                        backstress2, plastic_strain, cumeq, R)
   for i in 1:N
      dstress = randn(SymmetricTensor{2,3})
      dstrain = randn(SymmetricTensor{2,3})
      dbackstress1 = randn(SymmetricTensor{2,3})
      dbackstress2 = randn(SymmetricTensor{2,3})
      dplastic_strain = randn(SymmetricTensor{2,3})
      dcumeq = norm(dplastic_strain)
      dR = randn()
      dstate = AnotherState(dstress, dstrain, dbackstress1,
                           dbackstress2, dplastic_strain, dcumeq, dR)
      state = state + dstate
   end
   return state
end

println("Benchmark Chaboche State")
@btime bench_chaboche_style_state(N)
