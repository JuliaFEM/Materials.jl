# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

module Materials

include("types.jl")
export Material, IsotropicHooke, Plastic, VonMises, Model, create_material

include("plasticity.jl")
export yield_function, d_yield_function, equivalent_stress

include("solvers.jl")
export find_root, radial_return, array_to_tensor, tensor_to_array

include("response.jl")
export calc_response!

end
