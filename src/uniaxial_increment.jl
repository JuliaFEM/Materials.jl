# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

 function uniaxial_increment!(material, dstrain11,
        dt; dstrain=[dstrain11, -0.3*dstrain11, -0.3*dstrain11, 0.0, 0.0, 0.0],
        max_iter=50, norm_acc=1e-9)
    converged = false
    stress0 = tovoigt(material.variables.stress)
    for i=1:max_iter
        material.ddrivers.time = dt
        material.ddrivers.strain = fromvoigt(SymmetricTensor{2,3,Float64}, dstrain; offdiagscale=2.0)
        integrate_material!(material)
        stress = tovoigt(material.variables_new.stress)
        dstress = stress - stress0
        D = tovoigt(material.variables_new.jacobian)
        dstr = -D[2:end,2:end]\dstress[2:end]
        dstrain[2:end] .+= dstr
        norm(dstr) < norm_acc && (converged = true; break)
    end
    converged || error("No convergence in strain increment")
    return nothing
end
