# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Materials, FEMBase, Test

analysis, problem, element, bc_elements, ip = get_one_element_material_analysis(:IdealPlastic)
update!(element, "youngs modulus", 200.0e3)
update!(element, "poissons ratio", 0.3)
update!(element, "yield stress", 100.0)

times = [0.0, 1.0, 2.0, 3.0]
loads = [0.0, 1.0e-3, -1.0e-3, 1.0e-3]
loading = AxialStrainLoading(times, loads)
update_bc_elements!(bc_elements, loading)
analysis.properties.t1 = maximum(times)

run!(analysis)
s33 = [ip("stress", t)[3] for t in times]
s33_expected = [0.0, 100.0, -100.0, 100.0]
@test isapprox(s33, s33_expected; rtol=1.0e-2)
