module MaterialSimulators

using Materials, FEMBase, LinearAlgebra

# Material simulator to solve global system and run standard one element tests
include("mecamatso.jl")
export get_one_element_material_analysis, AxialStrainLoading, ShearStrainLoading, update_bc_elements!

# Material point simulator to study material behavior in single integration point
include("simulator.jl")
export Simulator

end
