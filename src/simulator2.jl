# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using Materials, LinearAlgebra

function uniaxial_increment!(material, dstrain11,
        dt; dstrain=[dstrain11, -0.3*dstrain11, -0.3*dstrain11, 0.0, 0.0, 0.0],
        max_iter=50, norm_acc=1e-9)
    material.dtime = dt
    converged = false
    for i=1:max_iter
        material.dstrain = dstrain
        integrate_material!(material)
        dstress = material.dstress
        D = material.jacobian
        dstr = -D[2:end,2:end]\dstress[2:end]
        dstrain[2:end] .+= dstr
        norm(dstr) < norm_acc && (converged = true; break)
    end
    converged || error("No convergence in strain increment")
    return nothing
end
