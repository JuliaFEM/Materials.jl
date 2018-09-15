# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using JuliaFEM, Materials, FEMBase, LinearAlgebra
using Materials: calculate_stress!

include("continuum.jl")

X = Dict(
    1 => [0.0, 0.0, 0.0],
    2 => [1.0, 0.0, 0.0],
    3 => [1.0, 1.0, 0.0],
    4 => [0.0, 1.0, 0.0],
    5 => [0.0, 0.0, 1.0],
    6 => [1.0, 0.0, 1.0],
    7 => [1.0, 1.0, 1.0],
    8 => [0.0, 1.0, 1.0])

body_element = Element(Hex8, (1, 2, 3, 4, 5, 6, 7, 8))
body_elements = [body_element]
update!(body_elements, "geometry", X)
update!(body_elements, "youngs modulus", 200.0e3)
update!(body_elements, "poissons ratio", 0.3)
update!(body_elements, "yield stress", 100.0)
update!(body_elements, "plastic strain", 0.0 => zeros(3,3))

bc_element_1 = Element(Poi1, (1,))
bc_element_2 = Element(Poi1, (2,))
bc_element_3 = Element(Poi1, (3,))
bc_element_4 = Element(Poi1, (4,))
bc_element_5 = Element(Poi1, (5,))
bc_element_6 = Element(Poi1, (6,))
bc_element_7 = Element(Poi1, (7,))
bc_element_8 = Element(Poi1, (8,))

bc_elements = [bc_element_1, bc_element_2, bc_element_3, bc_element_4,
               bc_element_5, bc_element_6, bc_element_7, bc_element_8]
update!(bc_elements, "geometry", X)
for element in (bc_element_1, bc_element_2, bc_element_3, bc_element_4)
    update!(element, "displacement 3", 0.0)
end
for element in (bc_element_5, bc_element_6, bc_element_7, bc_element_8)
    update!(element, "displacement 3", 0.0 => 0.0)
    update!(element, "displacement 3", 1.0 => 1.0e-3)
    update!(element, "displacement 3", 2.0 => -1.0e-3)
    update!(element, "displacement 3", 3.0 => 1.0e-3)
end

update!(bc_element_1, "displacement 1", 0.0)
update!(bc_element_1, "displacement 2", 0.0)
update!(bc_element_2, "displacement 2", 0.0)
update!(bc_element_4, "displacement 1", 0.0)

update!(bc_element_5, "displacement 1", 0.0)
update!(bc_element_5, "displacement 2", 0.0)
update!(bc_element_6, "displacement 2", 0.0)
update!(bc_element_8, "displacement 1", 0.0)

#update!(bc_element_5, "displacement 1", 0.0)
#update!(bc_element_5, "displacement 2", 0.0)
#update!(bc_element_5, "displacement 3", 0.0 => 0.0)
#update!(bc_element_5, "displacement 3", 1.0 => 1.0e-3)

# Initialize material model to integration points
for ip in get_integration_points(body_element)
    ip.fields["material"] = field(IdealPlastic(body_element, ip, 0.0))
    update!(ip, "stress", 0.0 => zeros(6))
    update!(ip, "strain", 0.0 => zeros(6))
end

body = Problem(Continuum3D, "1 element problem", 3)
bc = Problem(Dirichlet, "fix displacement", 3, "displacement")
add_elements!(body, body_elements)
add_elements!(bc, bc_elements)
analysis = Analysis(Nonlinear, "solve problem")
#xdmf = Xdmf("one_element_results_5"; overwrite=true)
#add_results_writer!(analysis, xdmf)
add_problems!(analysis, body, bc)
time_end = 3.0
dtime = 0.05

for problem in get_problems(analysis)
    initialize!(problem, analysis.properties.time)
end
update!(body_element, "displacement", analysis.properties.time => Dict(j => zeros(3) for j in 1:8))

while analysis.properties.time <= time_end
    analysis.properties.time += dtime
    @info("time = $(analysis.properties.time)")
    run!(analysis)
    # Postprocess
    for ip in get_integration_points(body_element)
        material = ip("material", analysis.properties.time)
        material.plastic_multiplier += material.dplastic_multiplier
        material.plastic_strain += material.dplastic_strain
        material.dplastic_multiplier = 0.0
        fill!(material.dplastic_strain, 0.0)

        D = zeros(6,6)
        S = zeros(6)
        calculate_stress!(material, body_element, ip, analysis.properties.time, dtime, D, S)
        gradu = body_element("displacement", ip, analysis.properties.time, Val{:Grad})
        strain = 0.5*(gradu + gradu')
        strain_vector = [strain[1,1], strain[2,2], strain[3,3],
                         2.0*strain[1,2], 2.0*strain[2,3], 2.0*strain[3,1]]
        update!(ip, "stress", analysis.properties.time => S)
        update!(ip, "strain", analysis.properties.time => strain_vector)
        update!(ip, "material matrix", analysis.properties.time => D)
        update!(ip, "plastic_multiplier", analysis.properties.time => copy(material.plastic_multiplier))
        update!(ip, "plastic strain", analysis.properties.time => copy(material.plastic_strain))

    end
    # update material internal parameters
end

#close(xdmf)

using Plots

ip = first(get_integration_points(body_element))
s11(t) = ip("stress", t)[1]
s22(t) = ip("stress", t)[2]
s33(t) = ip("stress", t)[3]
s12(t) = ip("stress", t)[4]
s23(t) = ip("stress", t)[5]
s31(t) = ip("stress", t)[6]
e11(t) = ip("strain", t)[1]
e22(t) = ip("strain", t)[2]
e33(t) = ip("strain", t)[3]
s(t) = ip("stress", t)
function vmis(t)
    s11, s22, s33, s12, s23, s31 = ip("stress", t)
    return sqrt(1/2*((s11-s22)^2 + (s22-s33)^2 + (s33-s11)^2 + 6*(s12^2+s23^2+s31^2)))
end

t = 0.0:dtime:time_end

u7 = zeros(3, length(t))
for (i,ti) in enumerate(t)
    u7[:,i] = body_element("displacement", ti)[7]
end

if true
    plot(e11.(t), s11.(t), label="\$\\sigma_{11}\$")
    plot!(e22.(t), s22.(t), label="\$\\sigma_{22}\$")
    plot!(e33.(t), s33.(t), label="\$\\sigma_{33}\$")
end

if false
    plot(t, u7', labels=["u1" "u2" "u3"])
end
