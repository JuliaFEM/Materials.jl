# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using ForwardDiff
using Tensors

"""
Equivalent tensile stress.

More info can be found from: https://en.wikipedia.org/wiki/Von_Mises_yield_criterion
    Section: Reduced von Mises equation for different stress conditions
"""
function equivalent_stress(stress)
    stress_dev = stress - 1/3 * trace(stress) * one(Tensor{2,3})
    return sqrt(3/2 * dcontract(stress_dev, stress_dev))
end

"""
    VonMises yield function
https://andriandriyana.files.wordpress.com/2008/03/yield_criteria.pdf
"""
function yield_function(stress, stress_y, ::Type{Val{:VonMises}})
    equivalent_stress(stress) - stress_y
end

"""
    d_yield_function(stress, ::Type{Val{:VonMises}}

Analytical gradient of Von Mises yield function
"""
function d_yield_function(stress, _, ::Type{Val{:VonMises}}) # Dirty hack here to avoid linter
    sigma_eq = equivalent_stress(stress)
    stress_dev = stress - 1/3 * trace(stress) * one(Tensor{2,3})
    3./2. * stress_dev / sigma_eq
end

# Aliases
function yield_function(stress, mat::VonMises)
    yield_function(stress, mat.yield_stress, Val{:VonMises})
end

function d_yield_function(stress, mat::VonMises)
    d_yield_function(stress, mat, Val{:VonMises})
end
