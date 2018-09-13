# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using FEMBase, LinearAlgebra

X = Dict(
    1 => [0.0, 0.0, 0.0],
    2 => [1.0, 0.0, 0.0],
    3 => [1.0, 1.0, 0.0],
    4 => [0.0, 1.0, 0.0],
    5 => [0.0, 0.0, 1.0],
    6 => [1.0, 0.0, 1.0],
    7 => [1.0, 1.0, 1.0],
    8 => [0.0, 1.0, 1.0])

u0 = Dict(j => zeros(3) for j in 1:8)
u1 = copy(u0)
u1[7] = [1.0, 0.0, 0.0]

element = Element(Hex8, (1, 2, 3, 4, 5, 6, 7, 8))
update!(element, "geometry", X)
update!(element, "displacement", 0.0 => u0)
update!(element, "displacement", 1.0 => u1)
update!(element, "youngs modulus", 100.0)
update!(element, "poissons ratio", 0.3)

function Base.run()
    time = 0.0
    time_end = 1.0
    dt = 0.5
    while time < time_end
        time += dt
        @info("time = $time")
        for ip in get_integration_points(element)
            gradu = element("displacement", ip, time, Val{:Grad})
            strain = 1/2*(gradu + gradu')
            E = element("youngs modulus", ip, time)
            nu = element("poissons ratio", ip, time)
            mu = E/(2.0*(1.0+nu))
            la = E*nu/((1.0+nu)*(1.0-2.0*nu))
            stress = la*tr(strain)*I + 2.0*mu*strain
            stress_dev = stress - 1/3*tr(stress)*I
            stress_v = sqrt(3/2*sum(stress_dev .* stress_dev))
            @info("Variables", ip.id, strain, stress, stress_dev, stress_v)
        end
    end
end

run()
