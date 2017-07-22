# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

module Materials

include("types.jl")
export Material, IsotropicHooke, Plastic, HyperElastic, add_property!

include("response.jl")
export calc_response!
end
