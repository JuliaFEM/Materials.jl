@material mymaterial
    @elasticity :isotropic "youngs modulus" "poissons ratio"
    @criterion :mises
    @flow :norton "K" "n"
    @kinematic :nonlinear "C1" "D1"
    @kinematic :jaska "C2" "D2" "gamma"
    @isotropic :nonlinear "R0" "Q" "b"
end
