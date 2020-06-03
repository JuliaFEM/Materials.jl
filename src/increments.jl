# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

# The skeleton of the optimizer is always the same, so we provide it as a
# higher-order function. The individual specific optimizers only need to
# define the "meat" of how to update `dstrain`.
"""
    optimize_dstrain!(material, dt, max_iter, tol, dstrain, update_dstrain!)

Find a compatible strain increment for `material`.

`dstrain` should initially be a relevant initial guess. At each iteration, it is
updated by the user-defined corrector `update_dstrain!`, whose call signature
must be:

    update_dstrain!(dstrain, dstress, D) -> convergence error measure

`dstress` is the difference between the stress state predicted by integrating
the material for one timestep, and the stress state stored in
`materials.variables.stress`.

`D` is the jacobian ∂σij/∂εkl (`material.variables_new.jacobian`), provided by
the material implementation.

The update is iterated at most `max_iter` times, until the error measure
returned by `update_dstrain!` falls below `tol`.

If `max_iter` is reached and the error measure is still `tol` or greater,
`ErrorException` is thrown.

Note the timestep is **not** committed; we call `integrate_material!`, but not
`update_material!`. Only `material.variables_new` is updated.
"""
function optimize_dstrain!(material, dt, max_iter, tol, dstrain, update_dstrain!)
    converged = false
    stress0 = tovoigt(material.variables.stress)  # observed
    for i=1:max_iter
        material.ddrivers.time = dt
        material.ddrivers.strain = fromvoigt(Symm2{Float64}, dstrain; offdiagscale=2.0)
        integrate_material!(material)
        stress = tovoigt(material.variables_new.stress)  # predicted
        dstress = stress - stress0
        D = tovoigt(material.variables_new.jacobian)
        e = update_dstrain!(dstrain, dstress, D)
        if e < tol
            converged = true
            break
        end
    end
    converged || error("No convergence in strain increment")
    return nothing
end

"""
    uniaxial_increment!(material, dstrain11, dt;
                        dstrain=[dstrain11, -0.3*dstrain11, -0.3*dstrain11, 0.0, 0.0, 0.0],
                        max_iter=50, norm_acc=1e-9)

Find a compatible strain increment for `material`.

The material state (`material.variables`) and the component 11 of the *strain*
increment are taken as prescribed. This routine computes the other components of
the strain increment. See `optimize_dstrain!`.
"""
function uniaxial_increment!(material, dstrain11, dt;
                             dstrain=[dstrain11, -0.3*dstrain11, -0.3*dstrain11, 0.0, 0.0, 0.0],
                             max_iter=50, norm_acc=1e-9)
    function update_dstrain!(dstrain, dstress, D)
        dstr = -D[2:end,2:end] \ dstress[2:end]
        dstrain[2:end] .+= dstr
        return norm(dstr)
    end
    optimize_dstrain!(material, dt, max_iter, norm_acc, dstrain, update_dstrain!)
    return nothing
end

"""
    biaxial_increment!(material, dstrain11, dstrain12, dt;
                       dstrain=[dstrain11, -0.3*dstrain11, -0.3*dstrain11, 0, 0, dstrain12],
                       max_iter=50, norm_acc=1e-9)

Find a compatible strain increment for `material`.

The material state (`material.variables`) and the components 11 and 12 of the
*strain* increment are taken as prescribed. This routine computes the other
components of the strain increment. See `optimize_dstrain!`.
"""
function biaxial_increment!(material, dstrain11, dstrain12, dt;
                            dstrain=[dstrain11, -0.3*dstrain11, -0.3*dstrain11, 0, 0, dstrain12],
                            max_iter=50, norm_acc=1e-9)
    function update_dstrain!(dstrain, dstress, D)
        dstr = -D[2:end-1,2:end-1] \ dstress[2:end-1]
        dstrain[2:end-1] .+= dstr
        return norm(dstr)
    end
    optimize_dstrain!(material, dt, max_iter, norm_acc, dstrain, update_dstrain!)
    return nothing
end

"""
    stress_driven_uniaxial_increment!(material, dstress11, dt;
                                      dstrain=[dstress11/200e3, -0.3*dstress11/200e3, -0.3*dstress11/200e3, 0.0, 0.0, 0.0],
                                      max_iter=50, norm_acc=1e-9)

Find a compatible strain increment for `material`.

The material state (`material.variables`) and the component 11 of the *stress*
increment are taken as prescribed. This routine computes the strain increment.
See `optimize_dstrain!`.
"""
function stress_driven_uniaxial_increment!(material, dstress11, dt;
                                           dstrain=[dstress11/200e3, -0.3*dstress11/200e3, -0.3*dstress11/200e3, 0.0, 0.0, 0.0],
                                           max_iter=50, norm_acc=1e-9)
    function update_dstrain!(dstrain, dstress, D)
        # Mutation here doesn't matter, since `dstress` is overwritten at the start of each iteration.
        # Note the lexical closure property gives us access to `dstress11` in this scope.
        dstress[1] -= dstress11
        dstr = -D \ dstress
        dstrain .+= dstr
        return norm(dstr)
    end
    optimize_dstrain!(material, dt, max_iter, norm_acc, dstrain, update_dstrain!)
    return nothing
end
