using Libdl
using LinearAlgebra

# These numbers are specific to the chosen umat
# TODO: Need to find the way how to set them and modify default state variables from user code
#       so that `reset_material!` and `update_material!` work correctly
NTENS = 6
NSTATV = 13
NPROPS = 1

# Documentation for every variable can be found in Abaqus doc about UMAT online.

"""
Variables updated by UMAT routine.
"""
@with_kw struct UmatVariableState <: AbstractMaterialState
    DDSDDE :: Array{Float64,2} = zeros(Float64, NTENS, NTENS)
    STRESS :: Array{Float64,1} = zeros(Float64, NTENS)
    STATEV :: Array{Float64,1} = zeros(Float64, NSTATV)
    SSE :: Array{Float64,1} = zeros(Float64, 1)
    SPD :: Array{Float64,1} = zeros(Float64, 1)
    SCD :: Array{Float64,1} = zeros(Float64, 1)
    RPL :: Array{Float64,1} = zeros(Float64, 1)
    DDSDDT :: Array{Float64,1} = zeros(Float64, NTENS)
    DRPLDE :: Array{Float64,1} = zeros(Float64, NTENS)
    DRPLDT :: Array{Float64,1} = zeros(Float64, 1)
    PNEWDT :: Array{Float64,1} = ones(Float64, 1)
end

"""
Material parameters in order that is specific to chosen UMAT.
"""
@with_kw struct UmatParameterState <: AbstractMaterialState
    PROPS :: Array{Float64,1} = zeros(Float64, NPROPS)
end

"""
Variables passed in for information.
These drive evolution of the material state.
"""
@with_kw struct UmatDriverState <: AbstractMaterialState
    STRAN :: Array{Float64,1} = zeros(Float64, NTENS)
    TIME :: Array{Float64,1} = zeros(Float64, 2)
    TEMP :: Float64 = zero(Float64)
    PREDEF :: Float64 = zero(Float64)
end

"""
Other Abaqus UMAT variables passed in for information.
"""
@with_kw struct UmatOtherState <: AbstractMaterialState
    CMNAME :: Cuchar = 'U'
    NDI :: Int64 = 3
    NSHR :: Int64 = 3
    NTENS :: Int64 = NTENS # NDI + NSHR
    NSTATV :: Int64 = NSTATV #zero(Int64)
    NPROPS :: Int64 = NPROPS #zero(Int64)
    COORDS :: Array{Float64,1} = zeros(Float64, 3)
    DROT :: Array{Float64,2} = I + zeros(Float64, 3, 3)
    CELENT :: Float64 = zero(Float64)
    DFGRD0 :: Array{Float64,2} = I + zeros(Float64, 3, 3)
    DFGRD1 :: Array{Float64,2} = I + zeros(Float64, 3, 3)
    NOEL :: Int64 = zero(Int64)
    NPT :: Int64 = zero(Int64)
    LAYER :: Int64 = zero(Int64)
    KSPT :: Int64 = zero(Int64)
    JSTEP :: Array{Float64,1} = zeros(Int64, 3)
    KSTEP :: Int64 = zero(Int64)
    KINC :: Int64 = zero(Int64)
end

"""
UMAT material structure.

`lib_path` is the path to the compiled shared library.
`behaviour` is the function name to call from the shared library
    generally this is compiler dependent, by default `umat_`.
    MFront Abaqus interface produces specific name, e.g. `ELASTICITY_3D`.
"""
@with_kw mutable struct UmatMaterial <: AbstractMaterial
    drivers :: UmatDriverState = UmatDriverState()
    ddrivers :: UmatDriverState = UmatDriverState()
    variables :: UmatVariableState = UmatVariableState()
    variables_new :: UmatVariableState = UmatVariableState()
    parameters :: UmatParameterState = UmatParameterState()
    dparameters :: UmatParameterState = UmatParameterState()
    
    umat_other :: UmatOtherState = UmatOtherState()

    lib_path :: String
    behaviour :: Symbol = :umat_
end

"""
Wrapper function to ccall the UMAT subroutine from the compiled shared library.
"""
function call_umat!(func_umat::Symbol, lib_path::String, STRESS,STATEV,DDSDDE,SSE,SPD,SCD,RPL,DDSDDT,DRPLDE,DRPLDT,
        STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME,NDI,NSHR,
        NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT,CELENT,DFGRD0,DFGRD1,
        NOEL,NPT,LAYER,KSPT,KSTEP,KINC)

    lib_umat = Libdl.dlopen(lib_path) # Open the library explicitly.
    sym_umat = Libdl.dlsym(lib_umat, func_umat) # Get a symbol for the umat function to call.

    ccall(sym_umat, Nothing, 
        (Ref{Float64},Ref{Float64},Ref{Float64},Ref{Float64},Ref{Float64},Ref{Float64},Ref{Float64},Ref{Float64},Ref{Float64},Ref{Float64},
        Ref{Float64},Ref{Float64},Ref{Float64},Ref{Float64},Ref{Float64},Ref{Float64},Ref{Float64},Ref{Float64},Ref{Cuchar},Ref{Int64},Ref{Int64},
        Ref{Int64},Ref{Int64},Ref{Float64},Ref{Int64},Ref{Float64},Ref{Float64},Ref{Float64},Ref{Float64},Ref{Float64},Ref{Float64},
        Ref{Int64},Ref{Int64},Ref{Int64},Ref{Int64},Ref{Int64},Ref{Int64}),
        STRESS,STATEV,DDSDDE,SSE,SPD,SCD,RPL,DDSDDT,DRPLDE,DRPLDT,
        STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME,NDI,NSHR,
        NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT,CELENT,DFGRD0,DFGRD1,
        NOEL,NPT,LAYER,KSPT,KSTEP,KINC)
    
    Libdl.dlclose(lib_umat) # Close the library explicitly.
    
    return Nothing
end

"""
Calls UMAT and updates MaterialVariableState writing the result to material.variables_new
"""
function integrate_material!(material::UmatMaterial)
    @unpack CMNAME,NDI,NSHR,NTENS,NSTATV,NPROPS,COORDS,DROT,CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,KSTEP,KINC = material.umat_other
    @unpack PROPS = material.parameters
    STRAN, TIME, TEMP, PREDEF = material.drivers.STRAN, material.drivers.TIME, material.drivers.TEMP, material.drivers.PREDEF
    DSTRAN, DTIME, DTEMP, DPRED = material.ddrivers.STRAN, material.drivers.TIME[1], material.ddrivers.TEMP, material.ddrivers.PREDEF
    @unpack STRESS,STATEV,DDSDDE,SSE,SPD,SCD,RPL,DDSDDT,DRPLDE,DRPLDT,PNEWDT = material.variables

    call_umat!(material.behaviour, material.lib_path, STRESS, STATEV, DDSDDE, SSE, SPD, SCD, RPL, DDSDDT, DRPLDE, DRPLDT,
        STRAN, DSTRAN, TIME, DTIME, TEMP, DTEMP, PREDEF, DPRED, CMNAME, NDI, NSHR,
        NTENS, NSTATV, PROPS, NPROPS, COORDS, DROT, PNEWDT, CELENT, DFGRD0, DFGRD1,
        NOEL, NPT, LAYER, KSPT, KSTEP, KINC)
    
    variables_new = UmatVariableState(STRESS=STRESS,STATEV=STATEV,DDSDDE=DDSDDE,SSE=SSE,SPD=SPD,SCD=SCD,RPL=RPL,DDSDDT=DDSDDT,DRPLDE=DRPLDE,DRPLDT=DRPLDT,PNEWDT=PNEWDT)
    material.variables_new = variables_new
end