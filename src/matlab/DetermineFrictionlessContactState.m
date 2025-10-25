function [ContactPairs, GKF, Residual] = DetermineFrictionlessContactState(FEMod, ContactPairs, ...
    Dt, PreDisp, GKF, Residual, Disp)

% --- Integration points (2x2 Gauss)
gp = 1/sqrt(3);
IntegralPoint = [-gp, -gp;
                  gp, -gp;
                  gp,  gp;
                 -gp,  gp];

% --- Contact search and friction factor
ContactPairs = ContactSearch(FEMod, ContactPairs, Disp, IntegralPoint);
FricFac = FEMod.FricFac;


nPairs = size(ContactPairs.SlaveSurf, 2);

% --- Loop over contact pairs
for i = 1:nPairs

    % Check for active contact
    if ContactPairs.CurMasterSurf(1, i) == 0
        continue; % No contact
    end

    % Case 1: first contact or frictionless contact
    if (FricFac == 0) || (ContactPairs.PreMasterSurf(1, i) == 0)
        ContactPairs.CurContactState(i) = 2; % Slip
        
        % [GKF, Residual, ContactPairs] = CalculateContactKandF_refac3(FEMod, ...
        %      ContactPairs, Dt, PreDisp, i, GKF, Residual, Disp, IntegralPoint);
        % 
        [GKF, Residual, ContactPairs] = CalculateFrictionlessContactKandF(FEMod, ...
            ContactPairs, Dt, PreDisp, i, GKF, Residual, Disp, IntegralPoint);

        continue;
    end
   
end
end
