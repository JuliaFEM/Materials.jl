# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/Materials.jl/blob/master/LICENSE

"""
The functions in this module are made to be able to easily simulate stress
states produced by some of the most common test machines.

Take for example the function `uniaxial_increment!`. In a push-pull machine
(with a smooth specimen), we know that the stress state is uniaxial (in the
measuring volume). Given the strain increment in the direction where the stress
is nonzero, we find a strain increment that produces zero stress in the other
directions. Similarly for the other functions.
"""
module Increments

import LinearAlgebra: norm
import Tensors: tovoigt, fromvoigt

import ..AbstractMaterial, ..integrate_material!
import ..Utilities: Symm2

export find_dstrain!, general_increment!, stress_driven_general_increment!,
       general_mixed_increment!,
       uniaxial_increment!, biaxial_increment!, stress_driven_uniaxial_increment!

"""
    find_dstrain!(material::AbstractMaterial, dstrain::AbstractVector{<:Real},
                  dt::Real, update_dstrain!::Function;
                  max_iter::Integer=50, tol::Real=1e-9)

Find a compatible strain increment for `material`.

Chances are you'll only need to call this low-level function directly if you
want to implement new kinds of strain optimizers. See `general_increment!`
and `stress_driven_general_increment!` for usage examples.

This is the skeleton of the optimizer. The individual specific optimizer
functions (`update_dstrain!)` only need to define how to update `dstrain`.
The skeleton itself isn't a Newton-Raphson root finder. It just abstracts away
the iteration loop, convergence checking and data plumbing, so it can be used,
among other kinds, to conveniently implement Newton-Raphson root finders.

The `dstrain` supplied to this function is the initial guess for the
optimization. At each iteration, it must be updated by the user-defined
corrector `update_dstrain!`, whose call signature is:

    update_dstrain!(dstrain::V, dstress::V, jacobian::AbstractArray{T})
        where V <: AbstractVector{T} where T <: Real
      -> err::Real

`dstrain` is the current value of the strain increment, in Voigt format.
Conversion to tensor format uses `offdiagscale=2.0`. The function must update
the Voigt format `dstrain` in-place.

`dstress = stress - stress0`, where `stress` is the stress state predicted by
integrating the material for one timestep of length `dt`, using the current
value of `dstrain` as a driving strain increment, and `stress0` is the stress
state stored in `materials.variables.stress`.

`jacobian` is ∂σij/∂εkl (`material.variables_new.jacobian`), as computed by the
material implementation. In many cases, the dstrain optimization can actually be
performed by a Newton-Raphson root finder, so we pass the jacobian to facilitate
writing the update formula for such a root finder.

The return value `err` must be an error measure (Real, >= 0).

The update is iterated at most `max_iter` times, until `err` falls below `tol`.

If `max_iter` is reached and the error measure is still `tol` or greater,
`ErrorException` is thrown.

To keep features orthogonal, the timestep is **not** committed automatically.
We call `integrate_material!`, but not `update_material!`. In other words,
we only update `material.variables_new`. To commit the timestep, call
`update_material!` after the optimizer is done.
"""
function find_dstrain!(material::AbstractMaterial, dstrain::AbstractVector{<:Real},
                       dt::Real, update_dstrain!::Function;
                       max_iter::Integer=50, tol::Real=1e-9)
    stress0 = tovoigt(material.variables.stress)  # stored
    T = typeof(dstrain[1])
    # @debug "---START---"
    for i=1:max_iter
        # @debug "$i, $dstrain, $stress0, $(material.variables.stress)"
        material.ddrivers.time = dt
        material.ddrivers.strain = fromvoigt(Symm2{T}, dstrain; offdiagscale=2.0)
        integrate_material!(material)
        stress = tovoigt(material.variables_new.stress)  # predicted
        dstress = stress - stress0
        jacobian = tovoigt(material.variables_new.jacobian)
        e = update_dstrain!(dstrain, dstress, jacobian)
        if e < tol
            return nothing
        end
    end
    error("No convergence in strain increment")
end

# --------------------------------------------------------------------------------

"""
    general_increment!(material::AbstractMaterial, dstrain_knowns::AbstractVector{Union{T, Missing}},
                       dstrain::AbstractVector{Union{T, Missing}}=dstrain_knowns,
                       max_iter::Integer=50, norm_acc::T=1e-9) where T <: Real

Find a compatible strain increment for `material`.

The material state (`material.variables`) and any non-`missing` components of
the *strain* increment `dstrain_knowns` are taken as prescribed. Any `missing`
components will be solved for.

This routine computes the `missing` components of the strain increment, such that
those components of the new stress state that correspond to the `missing` strain
increment components, remain at the old values stored in `material.variables.stress`.
(Often in practice, those old values are set to zero, allowing simulation of
uniaxial push-pull tests and similar.)

"New" stress state means the stress state after integrating the material by
one timestep of length `dt`.

The type of the initial guess `dstrain` is `AbstractVector{Union{T, Missing}}`
only so we can make it default to `dstrain_knowns`, which has that type. Any
`missing` components in the initial guess `dstrain` will be replaced by zeroes
before we invoke the solver.

See `find_dstrain!`.
"""
function general_increment!(material::AbstractMaterial,
                            dstrain_knowns::AbstractVector{<:Union{T, Missing}},
                            dt::Real,
                            dstrain::AbstractVector{<:Union{T, Missing}}=dstrain_knowns,
                            max_iter::Integer=50, norm_acc::T=1e-9) where T <: Real
    function validate_size(name::String, v::AbstractVector)
        if ndims(v) != 1 || size(v)[1] != 6
            error("""Expected a length-6 vector for $(name), got $(typeof(v)) with size $(join(size(v), "×"))""")
        end
    end
    validate_size("dstrain_knowns", dstrain_knowns)
    validate_size("dstrain", dstrain)
    dstrain_actual::AbstractVector{T} = T[((x !== missing) ? x : T(0)) for x in dstrain]
    dstrain_knowns_idxs = Integer[k for k in 1:6 if dstrain_knowns[k] !== missing]
    dstrain_unknown_idxs = setdiff(1:6, dstrain_knowns_idxs)
    if length(dstrain_unknown_idxs) == 0
        error("Optimizer needs at least one unknown dstrain component to solve for")
    end

    function update_dstrain!(dstrain::V, dstress::V, jacobian::AbstractArray{T}) where V <: AbstractVector{T} where T <: Real
        # See the stress-driven routine (`stress_driven_general_increment!`) for the general idea
        # of how this works. The differences to that algorithm are that:
        #
        #  - We update only components whose dstrain is not prescribed.
        #  - We want all corresponding components of dstress to converge to zero in the
        #    surrounding Newton-Raphson iteration.
        #
        dstrain_correction = (-jacobian[dstrain_unknown_idxs, dstrain_unknown_idxs]
                              \ dstress[dstrain_unknown_idxs])
        dstrain[dstrain_unknown_idxs] .+= dstrain_correction
        return norm(dstrain_correction)
    end
    find_dstrain!(material, dstrain_actual, dt, update_dstrain!, max_iter=max_iter, tol=norm_acc)
    dstrain[:] = dstrain_actual
    return nothing
end

"""
    stress_driven_general_increment!(material::AbstractMaterial,
                                     dstress_knowns::AbstractVector{<:Union{T, Missing}},
                                     dt::Real,
                                     dstrain::AbstractVector{T},
                                     max_iter::Integer=50, norm_acc::T=1e-9) where T <: Real

Find a compatible strain increment for `material`.

The material state (`material.variables`) and any non-`missing` components of
the *stress* increment `dstress_knowns` are taken as prescribed.

This routine computes a *strain* increment such that those components of the
new stress state, that correspond to non-`missing` components of `dstress_knowns`,
match those components of `material.variables.stress + dstress_knowns`.

For any `missing` components of `dstress_knowns`, the new stress state will match
the corresponding components of `material.variables.stress`. (So the `missing`
components act as if they were zero.)

*All* strain increment components will be solved for. If you need to prescribe
some of them, while also prescribing stresses, see `general_mixed_increment!`.

"New" stress state means the stress state after integrating the material by
one timestep of length `dt`.

`dstrain` is the initial guess for the strain increment.

See `find_dstrain!`.
"""
function stress_driven_general_increment!(material::AbstractMaterial,
                                          dstress_knowns::AbstractVector{<:Union{T, Missing}},
                                          dt::Real,
                                          dstrain::AbstractVector{T},
                                          max_iter::Integer=50, norm_acc::T=1e-9) where T <: Real
    function validate_size(name::String, v::AbstractVector)
        if ndims(v) != 1 || size(v)[1] != 6
            error("""Expected a length-6 vector for $(name), got $(typeof(v)) with size $(join(size(v), "×"))""")
        end
    end
    validate_size("dstress_knowns", dstress_knowns)
    validate_size("dstrain", dstrain)
    dstrain_actual::AbstractVector{T} = T[((x !== missing) ? x : T(0)) for x in dstrain]
    dstress_knowns_idxs = Integer[k for k in 1:6 if dstress_knowns[k] !== missing]

    function update_dstrain!(dstrain::V, dstress::V, jacobian::AbstractArray{T}) where V <: AbstractVector{T} where T <: Real
        # For the stress-driven correction, we have
        #
        #   dε = dε₀ + dεₐ
        #
        # where dε₀ is the dstrain currently suggested by the optimizer, and the adjustment dεₐ is
        #
        #   dεₐ = -(∂σ/∂ε)⁻¹ dσₑ
        #   dσₑ = dσ - dσₖ
        #
        # Here dσₑ is the effective stress increment, and dσₖ is the prescribed
        # (known) stress increment, which is zero for unknown dstress components.
        # As the Newton-Raphson iteration proceeds, dσₑ will converge to zero.
        #
        # Mutation of `dstress` doesn't matter, since `dstress` is freshly generated at each iteration.
        dstress[dstress_knowns_idxs] -= dstress_knowns[dstress_knowns_idxs]
        dstrain_correction = -jacobian \ dstress
        dstrain .+= dstrain_correction
        return norm(dstrain_correction)
    end
    find_dstrain!(material, dstrain_actual, dt, update_dstrain!, max_iter=max_iter, tol=norm_acc)
    dstrain[:] = dstrain_actual
    return nothing
end


"""
    general_mixed_increment!(material::AbstractMaterial,
                             dstrain_knowns::AbstractVector{<:Union{T, Missing}},
                             dstress_knowns::AbstractVector{<:Union{T, Missing}},
                             dt::Real,
                             dstrain::AbstractVector{<:Union{T, Missing}}=dstrain_knowns,
                             max_iter::Integer=50, norm_acc::T=1e-9) where T <: Real

Find a compatible strain increment for `material`. A combination of `general_increment!`
and `stress_driven_general_increment!`, which allows for loadings where some components
are strain-driven and some are stress-driven, such as the biaxial "bow-tie" and
"reverse bow-tie" loadings of:

    Corona, E., Hassan, T. and Kyriakides, S. (1996) On the Performance of
    Kinematic Hardening Rules in Predicting a Class of Biaxial Ratcheting
    Histories. International Journal of Plasticity, Vol 12, pp. 117--145.

See also:

    Bari, Shafiqul. (2001) Constitutive modeling for cyclic plasticity and ratcheting.
    Ph.D. thesis, North Carolina State University.

which compares the response of several different plasticity models under these loadings.

Each known component must be either strain-driven or stress-driven, not both.

The combination of the known components of dstrain and dstress must describe a
valid material state. Otherwise the optimizer may fail to converge, or return
a nonsensical solution.
"""
function general_mixed_increment!(material::AbstractMaterial,
                                  dstrain_knowns::AbstractVector{<:Union{T, Missing}},
                                  dstress_knowns::AbstractVector{<:Union{T, Missing}},
                                  dt::Real,
                                  dstrain::AbstractVector{<:Union{T, Missing}}=dstrain_knowns,
                                  max_iter::Integer=50, norm_acc::T=1e-9) where T <: Real
    function validate_size(name::String, v::AbstractVector)
        if ndims(v) != 1 || size(v)[1] != 6
            error("""Expected a length-6 vector for $(name), got $(typeof(v)) with size $(join(size(v), "×"))""")
        end
    end
    validate_size("dstrain_knowns", dstrain_knowns)
    validate_size("dstress_knowns", dstress_knowns)
    validate_size("dstrain", dstrain)
    dstrain_actual::AbstractVector{T} = T[((x !== missing) ? x : T(0)) for x in dstrain]
    dstrain_knowns_idxs = Integer[k for k in 1:6 if dstrain_knowns[k] !== missing]
    dstress_knowns_idxs = Integer[k for k in 1:6 if dstress_knowns[k] !== missing]
    dstrain_unknown_idxs = setdiff(1:6, dstrain_knowns_idxs)
    if length(dstrain_unknown_idxs) == 0
        error("Optimizer needs at least one unknown dstrain component to solve for")
    end

    # check that no component is being prescribed both ways
    let bad_idxs = intersect(dstrain_knowns_idxs, dstress_knowns_idxs)
        if length(bad_idxs) > 0
            plural = (length(bad_idxs) != 1) ? "s" : ""
            error("""Each known component must be either strain- or stress-driven, not both; please check the input for component$(plural) $(bad_idxs)""")
        end
    end

    function update_dstrain!(dstrain::V, dstress::V, jacobian::AbstractArray{T}) where V <: AbstractVector{T} where T <: Real
        # This update algorithm is a combination of those in `general_increment!`
        # and `stress_driven_general_increment!`.
        #
        #  - Like `general_increment!`, we update only components whose dstrain is not prescribed.
        #  - Like `stress_driven_general_increment!`, we allow for nonzero target dstress.
        #
        dstress[dstress_knowns_idxs] -= dstress_knowns[dstress_knowns_idxs]
        dstrain_correction = -(jacobian[dstrain_unknown_idxs, dstrain_unknown_idxs]
                               \ dstress[dstrain_unknown_idxs])
        dstrain[dstrain_unknown_idxs] .+= dstrain_correction

        return norm(dstrain_correction)
    end
    find_dstrain!(material, dstrain_actual, dt, update_dstrain!, max_iter=max_iter, tol=norm_acc)
    dstrain[:] = dstrain_actual
    return nothing
end

# --------------------------------------------------------------------------------

"""
    uniaxial_increment!(material::AbstractMaterial, dstrain11::Real, dt::Real;
                        dstrain::AbstractVector{<:Real}=[dstrain11, -0.3*dstrain11, -0.3*dstrain11, 0.0, 0.0, 0.0],
                        max_iter::Integer=50, norm_acc::Real=1e-9)

Find a compatible strain increment for `material`.

The material state (`material.variables`) and the component 11 of the *strain*
increment are taken as prescribed.

Convenience function; see `general_increment!`.

See `find_dstrain!`.
"""
function uniaxial_increment!(material::AbstractMaterial, dstrain11::Real, dt::Real;
                             dstrain::AbstractVector{<:Real}=[dstrain11, -0.3*dstrain11, -0.3*dstrain11, 0.0, 0.0, 0.0],
                             max_iter::Integer=50, norm_acc::Real=1e-9)
    dstrain_knowns = [dstrain11, missing, missing, missing, missing, missing]
    general_increment!(material, dstrain_knowns, dt, dstrain, max_iter, norm_acc)
    return nothing
end

"""
    biaxial_increment!(material::AbstractMaterial, dstrain11::Real, dstrain12::Real, dt::Real;
                       dstrain::AbstractVector{<:Real}=[dstrain11, -0.3*dstrain11, -0.3*dstrain11, 0, 0, dstrain12],
                       max_iter::Integer=50, norm_acc::Real=1e-9)

Find a compatible strain increment for `material`.

By "biaxial", we mean a stress state with one normal component and one shear
component.

The material state (`material.variables`) and the components 11 and 12 of the
*strain* increment are taken as prescribed.

Convenience function; see `general_increment!`.

See `find_dstrain!`.
"""
function biaxial_increment!(material::AbstractMaterial, dstrain11::Real, dstrain12::Real, dt::Real;
                            dstrain::AbstractVector{<:Real}=[dstrain11, -0.3*dstrain11, -0.3*dstrain11, 0, 0, dstrain12],
                            max_iter::Integer=50, norm_acc::Real=1e-9)
    dstrain_knowns = [dstrain11, missing, missing, missing, missing, dstrain12]
    general_increment!(material, dstrain_knowns, dt, dstrain, max_iter, norm_acc)
    return nothing
end

"""
    stress_driven_uniaxial_increment!(material::AbstractMaterial, dstress11::Real, dt::Real;
                                      dstrain::AbstractVector{<:Real}=[dstress11/200e3, -0.3*dstress11/200e3, -0.3*dstress11/200e3, 0.0, 0.0, 0.0],
                                      max_iter::Integer=50, norm_acc::Real=1e-9)

Find a compatible strain increment for `material`.

The material state (`material.variables`) and the component 11 of the *stress*
increment are taken as prescribed.

Convenience function; see `stress_driven_general_increment!`.

See `find_dstrain!`.
"""
function stress_driven_uniaxial_increment!(material::AbstractMaterial, dstress11::Real, dt::Real;
                                           dstrain::AbstractVector{<:Real}=[dstress11/200e3, -0.3*dstress11/200e3, -0.3*dstress11/200e3, 0.0, 0.0, 0.0],
                                           max_iter::Integer=50, norm_acc::Real=1e-9)
    dstress_knowns = [dstress11, missing, missing, missing, missing, missing]
    stress_driven_general_increment!(material, dstress_knowns, dt, dstrain, max_iter, norm_acc)
    return nothing
end

end
