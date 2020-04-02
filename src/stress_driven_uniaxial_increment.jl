# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

using NLsolve

function stress_driven_uniaxial_increment!(material, dstrain11, dtime; 
        dstrain=[dstrain11, -0.3*dstrain11, -0.3*dstrain11, 0.0, 0.0, 0.0],
        max_iter=50, norm_acc=1e-9)
    converged = false
    stress0 = tovoigt(material.variables.stress)
    println("************************************")
    println("stress0:    ", typeof(stress0), ", size; ", size(stress0), "\n", dstrain)
    println("------------------------------------")
    for i=1:max_iter
        #dstrainguess = [dstrain11, -0.3*dstrain11, -0.3*dstrain11, 0.0, 0.0, 0.0]
        #f!(creepres, tosolve) = creepresidual!(creepres, tosolve, material, dtime, stress0)
        #dstrain = nlsolve( f!, dstrainguess ).zero # , autodiff = :forward 
        
        material.ddrivers.time = dtime
        material.ddrivers.strain = fromvoigt(SymmetricTensor{2,3,Float64}, dstrain; offdiagscale=2.0)
        integrate_material!(material)
        stress = tovoigt(material.variables_new.stress)
        dstress = stress - stress0
        
        dstress11 = dstress[1]
        stresslimit = stress[1]

        if stresslimit >= material.parameters.yield_stress
            dstress[1:end] .= 0.0
            creepstress = [material.parameters.yield_stress, 0.0, 0.0, 0.0, 0.0, 0.0]
            newstress = fromvoigt(SymmetricTensor{2,3,Float64}, creepstress; offdiagscale=2.0)

            E       = material.parameters.youngs_modulus
            nu      = material.parameters.poissons_ratio

            dstrain = [(1/E)*dstress[1], -(nu/E)*dstress[1], -(nu/E)*dstress[1], 0.0, 0.0, 0.0] .* 0.0
            material.drivers.strain  = fromvoigt(SymmetricTensor{2,3,Float64}, dstrain; offdiagscale=2.0)
            material.ddrivers.strain = fromvoigt(SymmetricTensor{2,3,Float64}, dstrain; offdiagscale=2.0)


            println("------------------------------------")
            println("dstrain(1): ", typeof(dstrain), ", size; ", size(dstrain), "\n", dstrain)
            println("------------------------------------")
            
            material.variables_new = IdealPlasticVariableState(
                stress          = newstress,
                plastic_strain  = material.variables_new.plastic_strain,
                cumeq           = material.variables_new.cumeq,
                jacobian        = material.variables_new.jacobian)

        else
            dstress[2:end] .= 0.0
        end

        D = tovoigt(material.variables_new.jacobian)
        dstr = -D[2:end,2:end]\dstress[2:end]
        dstrain[2:end] .+= dstr
        
        stress = tovoigt(material.variables_new.stress)

        println("stress:     ", typeof(stress), ", size; ", size(stress), "\n", stress)
        println("dstress:    ", typeof(dstress), ", size; ", size(dstress), "\n", dstress)
        println("dstrain:    ", typeof(dstrain), ", size; ", size(dstrain), "\n", dstrain)
        println("------------------------------------")

        norm(dstr) < norm_acc && (converged = true; break)
    end

    converged || error("No convergence in strain increment")
    return nothing
end

function creepresidual!(creepres, tosolve, material, dtime, stress0)

    #guess = [dstrain11, -0.3*dstrain11, -0.3*dstrain11, 0.0, 0.0, 0.0]
    #f!() = residual!(material, dtime, tosolve)
    #dstrain = nlsolve( f!, guess, autodiff = :forward ).zero

    dstrain::Array{Float64,1}   = tosolve
    tempmaterial                = material

    tempmaterial.ddrivers.time   = dtime
    tempmaterial.ddrivers.strain = fromvoigt(SymmetricTensor{2,3,Float64}, dstrain; offdiagscale=2.0)
    
    integrate_material!(tempmaterial)
    
    #stress0     = tovoigt(tempmaterial.variables.stress)
    stress      = tovoigt(tempmaterial.variables_new.stress)
    dstress     = stress - stress0
    creepstress = tempmaterial.parameters.yield_stress
    creepvector = [creepstress, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    creepres    = dstress - creepvector
    
    return creepres, tosolve
end