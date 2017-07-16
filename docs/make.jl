# This file is a part of project JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Documenter
using Materials

makedocs(
    modules = [Materials],
    checkdocs = :all,
    strict = true)
