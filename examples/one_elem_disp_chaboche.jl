# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using JuliaFEM, FEMBase, LinearAlgebra

include("continuum.jl")
include("chaboche.jl")

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
update!(body_elements, "K_n", 100.0)
update!(body_elements, "n_n", 10.0)
update!(body_elements, "C_1", 10000.0)
update!(body_elements, "D_1", 100.0)
update!(body_elements, "C_2", 50000.0)
update!(body_elements, "D_2", 1000.0)
update!(body_elements, "Q", 50.0)
update!(body_elements, "b", 0.1)
update!(body_elements, "yield stress", 100.0)

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
    update!(element, "displacement 3", 1.0 => 5.0e-3)
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
    update!(ip, "yield stress", 0.0 => body_element("yield stress", ip, 0.0))
    update!(ip, "plastic strain", 0.0 => zeros(6))
    update!(ip, "stress", 0.0 => zeros(6))
    update!(ip, "strain", 0.0 => zeros(6))
    update!(ip, "backstress 1", 0.0 => zeros(6))
    update!(ip, "backstress 2", 0.0 => zeros(6))
    update!(ip, "cumulative equivalent plastic strain", 0.0 => 0.0)
    ip.fields["material"] = field(Chaboche(body_element, ip, 0.0))
end

body = Problem(Continuum3D, "1 element problem", 3)
bc = Problem(Dirichlet, "fix displacement", 3, "displacement")
add_elements!(body, body_elements)
add_elements!(bc, bc_elements)
analysis = Analysis(Nonlinear, "solve problem")
# xdmf = Xdmf("results"; overwrite=true)
# add_results_writer!(analysis, xdmf)
add_problems!(analysis, body, bc)
time_end = 1.0
dtime = 0.05

for problem in get_problems(analysis)
    initialize!(problem, analysis.properties.time)
end

while analysis.properties.time < time_end
    analysis.properties.time += dtime
    update!(body_element, "displacement", analysis.properties.time => Dict(j => zeros(3) for j in 1:8))
    @info("time = $(analysis.properties.time)")
    run!(analysis)
    for ip in get_integration_points(body_element)
        material = ip("material", analysis.properties.time)
        update!(ip, "stress", analysis.properties.time => copy(material.stress))
        update!(ip, "plastic strain", analysis.properties.time => copy(material.plastic_strain))
        update!(ip, "cumulative equivalent plastic strain", analysis.properties.time => copy(material.cumulative_equivalent_plastic_strain))
        update!(ip, "backstress 1", analysis.properties.time => copy(material.backstress1))
        update!(ip, "backstress 2", analysis.properties.time => copy(material.backstress2))
        update!(ip, "yield stress", analysis.properties.time => copy(material.yield_stress))
    end
    # update material internal parameters
end

# close(xdmf)

using Plots
if true
    ip1 = first(get_integration_points(body_element))
    t = range(0, stop=time_end, length=Int(time_end/dtime)+1)
    s11(t) = ip1("stress", t)[1]
    s22(t) = ip1("stress", t)[2]
    s33(t) = ip1("stress", t)[3]
    s12(t) = ip1("stress", t)[4]
    s23(t) = ip1("stress", t)[5]
    s31(t) = ip1("stress", t)[6]
    e11(t) = ip1("strain", t)[1]
    e22(t) = ip1("strain", t)[2]
    e33(t) = ip1("strain", t)[3]
    s(t) = ip1("stress", t)
    function vmis(t)
        s11, s22, s33, s12, s23, s31 = ip1("stress", t)
        return sqrt(1/2*((s11-s22)^2 + (s22-s33)^2 + (s33-s11)^2 + 6*(s12^2+s23^2+s31^2)))
    end
    plot(e11.(t), s11.(t), label="\$\\sigma_{11}\$")
    plot!(e22.(t), s22.(t), label="\$\\sigma_{22}\$")
    plot!(e33.(t), s33.(t), label="\$\\sigma_{33}\$")
    # labels = ["s11" "s22" "s33" "s12" "s23" "s31"]
    # plot(t, s11, title="stress at integration point 1", label="s11")
    # plot!(t, s22, label="s22")
    # plot!(t, s33, label="s33")
    # plot!(t, s12, label="s12")
    # plot!(t, s23, label="s23")
    # plot!(t, s31, label="s31")
end
