# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

function tensor_to_array(A)
    vec([A[1,1], A[2,2], A[3,3], A[1,2], A[2,3], A[1,3]])
end

function array_to_tensor(A)
    Tensor{2,3}([A[1] A[4] A[6];
                 A[4] A[2] A[5];
                 A[6] A[5] A[3]])
end

"""
Creating functions for newton: xₙ₊₁ = xₙ - df⁻¹ * f and initial values
"""
function find_root(f, df, x; max_iter=50, norm_acc=1e-9)
    converged = false
    for i=1:max_iter
        dx = -df(x) \ f(x)
        x += dx
        norm(dx) < norm_acc && (converged = true; break)
    end
    converged || error("No convergence in radial return!")
    return x
end

"""
"""
function radial_return(input, params::Dict{AbstractString, Any})

    dstress = array_to_tensor(input)
    lambda = input[end]

    D = params["D"]
    yield_function = params["yield_function"]
    d_yield_function = params["d_yield_function"]
    stress_base = params["init_stress"]
    dstrain = params["dstrain"]

    # Stress rate and total strain
    stress_tot = stress_base + dstress

    # Calculating plastic strain rate
    dstrain_p = lambda * d_yield_function(stress_tot)

    # Calculating equations
    function_1 = dstress - dcontract(D, dstrain - dstrain_p)
    function_2 = yield_function(stress_tot)
    [tensor_to_array(function_1); function_2]
end

# """
# """
# function ideal_plasticity!(stress_new, stress_last, dstrain_vec, pstrain, D, params, Dtan, yield_surface_, time, dt, type_)
#     # Test stress
#     dstress = vec(D * dstrain_vec)
#     stress_trial = stress_last + dstress
#     stress_y = params["yield_stress"]

#     yield_curr = x -> yield_function(x, stress_y, yield_surface_, type_)

#     # Calculating and checking for yield
#     yield = yield_curr(stress_trial)
#     if isless(yield, 0.0)

#         stress_new[:] = stress_trial[:]
#         Dtan[:,:] = D[:,:]
#     else
#         # Creating functions for newton: xₙ₊₁ = xₙ - df⁻¹ \ f and initial values
#         f = stress_ -> radial_return(stress_, dstrain_vec, D, stress_y, stress_last, yield_surface_, type_)
#         df = x -> ForwardDiff.jacobian(f, x)

#         # Calculating root (two options)
#         vals = [vec(stress_trial - stress_last); 0.0]

#         results = find_root!(f, df, vals)

#         # extracting results
#         dstress = results[1:end-1]
#         plastic_multiplier = results[end]

#         # Updating stress
#         stress_new[:] = stress_last + dstress


#         # Calculating plastic strain
#         dfds_ = x -> ForwardDiff.gradient(yield_curr, x)
#         dep = plastic_multiplier * dfds_(vec(stress_new))

#         # Equations for consistent tangent matrix can be found from:
#         # http://homes.civil.aau.dk/lda/continuum/plast.pdf
#         # equations: 152 & 153
#         D2g = x -> ForwardDiff.hessian(yield_curr, x)
#         Dc = (D^-1 + plastic_multiplier * D2g(stress_new))^-1
#         dfds = dfds_(stress_new)
#         Dtan[:,:] = Dc - (Dc * dfds * dfds' * Dc) / (dfds' * Dc * dfds)[1]
#         pstrain[:] = plastic_multiplier * dfds
#     end
# end