# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Materials, FEMBase, Test

analysis, problem, element, bc_elements, ip = get_material_analysis(:IdealPlastic)
update!(element, "youngs modulus", 200.0e3)
update!(element, "poissons ratio", 0.3)
update!(element, "yield stress", 100.0)
for element in bc_elements
    update!(element, "fixed displacement 3", 0.0 => 0.0)
    update!(element, "fixed displacement 3", 1.0 => 1.0e-3)
    update!(element, "fixed displacement 3", 2.0 => -1.0e-3)
    update!(element, "fixed displacement 3", 3.0 => 1.0e-3)
end
run!(analysis)
s33 = [ip("stress", t)[3] for t in 0.0:0.1:1.0]
s33_expected = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
@test isapprox(s33, s33_expected; rtol=1.0e-2)
