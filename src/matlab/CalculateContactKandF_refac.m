function [GKF, Residual, ContactPairs] = CalculateContactKandF_refac(FEMod, ContactPairs, Dt, ...
    PreDisp, i, GKF, Residual, Disp, IntegralPoint)
% === Obtain contact stiffness and contact force ===
% This refactor preserves the original algorithm and numeric behaviour.
%
% Note: ContactPairs is a struct-of-vectors: fields are arrays/matrices,
%       use ContactPairs.Field(:, i) for 2xN / matrix fields and
%       ContactPairs.Field(i) for scalar/vector fields.
% 
FricFac = FEMod.FricFac;
 
if ContactPairs.CurContactState(i) == 1  % --- Stick contact state ---
    [GKF, Residual, ContactPairs] = CalculateContactKandF_stick(FEMod, ContactPairs, Dt, ...
    PreDisp, i, GKF, Residual, Disp, IntegralPoint);
 
elseif ContactPairs.CurContactState(i) == 2  % --- Slip contact state ---
     [GKF, Residual, ContactPairs] = CalculateContactKandF_slip(FEMod, ContactPairs, Dt, PreDisp, i, GKF, Residual, Disp, IntegralPoint);
end

end
