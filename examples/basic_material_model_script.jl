using LinearAlgebra
using Einsum
using Test
using NLsolve
using Plots
pyplot() # OLD BACKEND: plotly()

# Tensors
I_ = Matrix(1.0I,3,3) # Second order identity tensor
II = zeros(3,3,3,3)
@einsum II[i,j,k,l] = 0.5*(I_[i,k]*I_[j,l] + I_[i,l]*I_[j,k]) # Fourth order symmetric identity tensor
IxI = zeros(3,3,3,3)
@einsum IxI[i,j,k,l] = I_[i,j]*I_[k,l] # "Trace" tensor
P = II - 1/3*IxI # Deviatoric projection tensor

# Functions
function double_contraction(x::AbstractArray{<:Number,2},y::AbstractArray{<:Number,2})
    return sum(x.*y)
end

function double_contraction(x::AbstractArray{<:Number,4},y::AbstractArray{<:Number,2})
    retval = zeros(3,3)
    @einsum retval[i,j] = x[i,j,k,l]*y[k,l]
    return retval
end

A = rand(3,3) # Create Random second order tensor
A += A' # Symmetrize it

@test isapprox(double_contraction(II, A), A)
@test isapprox(double_contraction(IxI, A), I_*tr(A))

function deviator(x::AbstractArray{<:Number,2})
    s = zeros(3,3)
    @einsum s[i,j] = P[i,j,k,l]*x[k,l]
    return s
end

function von_mises_stress(stress::AbstractArray{<:Number,2})
    s = deviator(stress)
    return sqrt(3/2*double_contraction(s,s))
end

S = [100 0 0; 0 0 0; 0 0 0]
@test isapprox(von_mises_stress(S), 100)

### Material parameters ###
# Isotropic elasticity: \dot{sigma} = \mathcal{C}:(\dot{\varepsilon}_{tot} - \dot{\varepsilon}_{pl})
E = 210000.0 # Young's modulus
nu = 0.3 # Poisson's ratio
K = E/(3*(1-2*nu)) # Bulk modulus
G = E/(2*(1+nu)) # Shear modulus
C = K*IxI + 2*G*P # Elasticity Tensor \mathcal{C}

@test isapprox(double_contraction(C, [0.001 0 0; 0 -nu*0.001 0; 0 0 -nu*0.001]), [E*0.001 0 0; 0 0 0; 0 0 0])

# Non-linear isotropic hardening: \dot{R} = b(Q-R)\dot{p}
# where \dot{p} = \sqrt{2/3 \dot{\varepsilon}_{pl}:\dot{\varepsilon}_{pl}}} - equivalent plastic strain rate
R0 = 100.0 # Initial proportionality limit
Q = 50.0 # Hardening magnitude
b = 0.1 # Hardening rate

# Non-linear kinematic hardening: \dot{X}_i = 2/3C_i\dot{p}(n - \frac{3D_i}{2C_i}X_i)
# where n = \frac{\partial f}{\partial \sigma} - plastic strain direction
# and X = \sum_{i=1}^N X_i
C_1 = 10000.0 # Slope parameter 1
D_1 = 100.0 # Rate parameter 1

C_2 = 50000.0 # Slope parameter 2
D_2 = 1000.0 # Rate parameter 2

# Viscoplasticity: Norton viscoplastic potential \phi = \frac{K_n}{n_n+1}\left( \frac{f}{K_n} \right)^{n_n+1}
# \dot{\varepsilon}_{pl} = \frac{\partial \phi}{\partial \sigma} = \frac{\partial \phi}{\partial f}\frac{\partial f}{\partial \sigma}
# => \dot{p} = \frac{\partial \phi}{\partial f} = \left( \frac{f}{K_n} \right)^n_n
# => n = \frac{\partial f}{\partial \sigma}
# => \dot{\varepsilon}_{pl} = \dot{p} n
K_n = 100.0 # Drag stress
n_n = 3.0 # Viscosity exponent

# Initialize variables
sigma = zeros(3,3)
R = R0
X_1 = zeros(3,3)
X_2 = zeros(3,3)
varepsilon_pl = zeros(3,3)
varepsilon_el = zeros(3,3)
t = 0.0

# Determine loading sequence
varepsilon_a = 0.01 # Strain amplitude
#varepsilon_tot(t) = sin(t)*[varepsilon_a 0 0; 0 -nu*varepsilon_a 0; 0 0 -nu*varepsilon_a]
dt = 0.01 # Time step
T0 = 1.0
T = 10.0 # Time span
# varepsilon_tot(t) = t/T*[varepsilon_a 0 0; 0 -nu*varepsilon_a 0; 0 0 -nu*varepsilon_a]
function varepsilon_tot(t)
    if t<T0
        return t/T0*[varepsilon_a 0 0; 0 -nu*varepsilon_a 0; 0 0 -nu*varepsilon_a]
    else
        return [varepsilon_a 0 0; 0 -nu*varepsilon_a 0; 0 0 -nu*varepsilon_a]
    end
end


# Initialize result storage
ts = [t]
sigmas = [sigma]
Rs = [R]
X_1s = [X_1]
X_2s = [X_2]
varepsilon_pls = [varepsilon_pl]
varepsilon_els = [varepsilon_el]

# Time integration
while t < T
    global t, sigma, R, X_1, X_2, varepsilon_pl, varepsilon_el, ts, sigmas, Rs, X_1s, X2_s, varepsilon_pls, varepsilon_els
    global C, K_n, n_n, C_1, D_1, C_2, D_2, Q, b
    # Store initial state
    sigma_n = sigma
    R_n = R
    X_1n = X_1
    X_2n = X_2
    varepsilon_pln = varepsilon_pl
    varepsilon_eln = varepsilon_el
    t_n = t

    # Increments
    t = t + dt
    dvarepsilon_tot = varepsilon_tot(t) - varepsilon_tot(t_n)
    # Elastic trial
    sigma_tr = sigma_n + double_contraction(C, dvarepsilon_tot)
    # Check for yield
    f_tr = von_mises_stress(sigma_tr - X_1 - X_2) - R
    println("***************************************")
    if f_tr <= 0 # Elastic step
        # Update variables
        println("Elastic step!")
        sigma = sigma_tr
        varepsilon_el += dvarepsilon_tot
    else # Viscoplastic step
        println("Viscoplastic step!")
        function g!(F, x) # System of non-linear equations
            sigma = reshape(x[1:9], 3,3)
            R = x[10]
            X_1 = reshape(x[11:19], 3,3)
            X_2 = reshape(x[20:28], 3,3)
            dotp = ((von_mises_stress(sigma - X_1 - X_2) - R)/K_n)^n_n
            dp = dotp*dt
            s = deviator(sigma - X_1 - X_2)
            n = 3/2*s/von_mises_stress(sigma - X_1 - X_2)
            dvarepsilon_pl = dp*n
            f1 = vec(sigma_n - sigma + double_contraction(C, dvarepsilon_tot - dvarepsilon_pl))
            f2 = R_n - R + b*(Q-R)*dp
            f3 = vec(X_1n - X_1 + 2/3*C_1*dp*(n - 3*D_1/(2*C_1)*X_1))
            f4 = vec(X_2n - X_2 + 2/3*C_2*dp*(n - 3*D_2/(2*C_2)*X_2))
            F[:] = vec([f1; f2; f3; f4])
        end
        x0 = vec([vec(sigma_tr); R; vec(X_1); vec(X_2)])
        F = similar(x0)
        res = nlsolve(g!, x0)
        x = res.zero
        sigma = reshape(x[1:9],3,3)
        R = x[10]
        X_1 = reshape(x[11:19], 3,3)
        X_2 = reshape(x[20:28], 3,3)
        dotp = ((von_mises_stress(sigma - X_1 - X_2) - R)/K_n)^n_n
        dp = dotp*dt
        s = deviator(sigma - X_1 - X_2)
        n = 3/2*s/von_mises_stress(sigma - X_1 - X_2)
        dvarepsilon_pl = dp*n
        varepsilon_pl += dvarepsilon_pl
        varepsilon_el += dvarepsilon_tot - dvarepsilon_pl
    end

    # Store variables
    push!(ts, t)
    push!(sigmas, sigma)
    push!(Rs, R)
    push!(X_1s, X_1)
    push!(X_2s, X_2)
    push!(varepsilon_pls, varepsilon_pl)
    push!(varepsilon_els, varepsilon_el)
end

qs = [von_mises_stress(sigma_i) for sigma_i in sigmas]
ps = [tr(sigma_i)/3 for sigma_i in sigmas]
xs = [von_mises_stress(X_1s[i] + X_2s[i]) for i in 1:length(ts)]
plot(ps, qs, label="Stress")
plot!(ps, xs+Rs, label="Static yield surface")
xlabel!("Hydrostatic stress")
ylabel!("Von Mises stress")
