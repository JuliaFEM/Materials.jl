using Einsum
using Base.Test

# Tensors
I = eye(3) # Second order identity tensor
II = zeros(3,3,3,3)
@einsum II[i,j,k,l] = I[i,k]*I[j,l] # Fourth order symmetric identity tensor
IxI = zeros(3,3,3,3)
@einsum IxI[i,j,k,l] = I[i,j]*I[k,l] # "Trace" tensor
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
@test isapprox(double_contraction(IxI, A), eye(3)*trace(A))

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
n_n = 10.0 # Viscosity exponent

# Initialize variables
sigma = zeros(3,3)
R = R0
X_1 = zeros(3,3)
X_2 = zeros(3,3)
varepsilon_pl = zeros(3,3)
