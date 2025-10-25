function [ContactPairs, GKF, Residual] = DetermineContactState(FEMod, ContactPairs, ...
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
    % 
    % --- Case 2: possible stick/slip contact ---
    CurIP = IntegralPoint(ContactPairs.SlaveIntegralPoint(i), :)';
    [Na, N1a, N2a] = GetSurfaceShapeFunction(CurIP(1), CurIP(2));

    % Slave surface coordinates
    [CurSlaveSurfXYZ, ~] = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs.SlaveSurf(:, i));
    Cur_x1 = sum(Na .* CurSlaveSurfXYZ, 1)';

    % Master surface (previous)
    [Nb, ~, ~] = GetSurfaceShapeFunction(ContactPairs.rp(i), ContactPairs.sp(i));
    [CurMasterSurfXYZ_p, ~] = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs.PreMasterSurf(:, i));
    Cur_x2_p = sum(Nb .* CurMasterSurfXYZ_p, 1)';

    % Relative motion and projection
    gs = Cur_x2_p - Cur_x1;
    tv = ContactPairs.pc(i) * gs;

    % Current normal
    Cur_N1Xa = sum(N1a .* CurSlaveSurfXYZ, 1)';
    Cur_N2Xa = sum(N2a .* CurSlaveSurfXYZ, 1)';
    Cur_n = cross(Cur_N1Xa, Cur_N2Xa);
    Cur_n = Cur_n / norm(Cur_n);

    % Tangential/normal trial components
    tn_trial = abs(tv' * Cur_n);
    tt_trial = sqrt(norm(tv)^2 - tn_trial^2);

    % Slip/stick criterion
    fai = tt_trial - FricFac * tn_trial;
    if fai < 0
        ContactPairs.CurContactState(i) = 1; % Stick
    else
        ContactPairs.CurContactState(i) = 2; % Slip
    end
    
    % % Update stiffness and force
    [GKF, Residual, ContactPairs] = CalculateContactKandF_refac3(FEMod, ...
         ContactPairs, Dt, PreDisp, i, GKF, Residual, Disp, IntegralPoint);
end
end
