# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Materials, FEMBase, LinearAlgebra

# Standard simulation of ideal plastic material model

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

analysis.properties.t1 = 3.0
analysis.properties.extrapolate_initial_guess = false
run!(analysis)

s11(t) = ip("stress", t)[1]
s22(t) = ip("stress", t)[2]
s33(t) = ip("stress", t)[3]
s12(t) = ip("stress", t)[4]
s23(t) = ip("stress", t)[5]
s31(t) = ip("stress", t)[6]

e11(t) = ip("strain", t)[1]
e22(t) = ip("strain", t)[2]
e33(t) = ip("strain", t)[3]
e12(t) = ip("strain", t)[4]
e23(t) = ip("strain", t)[5]
e31(t) = ip("strain", t)[6]

using Plots, Test
t = 0.0:0.1:3.0
@test isapprox(maximum(e33.(t)), 0.001)
@test isapprox(minimum(e33.(t)), -0.001)
@test isapprox(maximum(s33.(t)), 100.0)
@test isapprox(minimum(s33.(t)), -100.0)
plot(e11.(t), s11.(t), label="\$\\sigma_{11}\$")
plot!(e22.(t), s22.(t), label="\$\\sigma_{22}\$")
plot!(e33.(t), s33.(t), label="\$\\sigma_{33}\$")
title!("Stress-strain curve of idealplastic material model, uniaxial strain")
ylabel!("Stress [MPa]")
xlabel!("Strain [str]")
savefig(joinpath("one_element_ideal_plastic/uniaxial_strain.svg"))
