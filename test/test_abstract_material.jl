# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Test

struct MyMaterial <: AbstractMaterial
    E :: Float64
end

function MyMaterial()
    error("MyMaterial needs at least `E` defined.")
end

mat = Material(MyMaterial, (E=210.0,))
@test mat.properties.E == 210.0
