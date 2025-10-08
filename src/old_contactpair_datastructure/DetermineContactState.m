% === Determine the contact state, then calculate contact stiffness and contact force ===
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

% --- Loop over contact pairs
for i = 1:size(ContactPairs, 1)

    % Check for active contact
    if ContactPairs(i).CurMasterSurf(1) == 0
        continue; % No contact
    end

    % Case 1: first contact or frictionless contact
    if (FricFac == 0) || (ContactPairs(i).PreMasterSurf(1) == 0)
        ContactPairs(i).CurContactState = 2; % Slip

        [GKF, Residual, ContactPairs] = CalculateContactKandF(FEMod, ...
            ContactPairs, Dt, PreDisp, i, GKF, Residual, Disp, IntegralPoint);

        continue;
    end

    % --- Case 2: possible stick/slip contact ---
    CurIP = IntegralPoint(ContactPairs(i).SlaveIntegralPoint, :)';
    [Na, N1a, N2a] = GetSurfaceShapeFunction(CurIP(1), CurIP(2));

    [CurSlaveSurfXYZ, ~] = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs(i).SlaveSurf);
    Cur_x1 = sum(Na .* CurSlaveSurfXYZ, 1)';

    [Nb, ~, ~] = GetSurfaceShapeFunction(ContactPairs(i).rp, ContactPairs(i).sp);
    [CurMasterSurfXYZ_p, ~] = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs(i).PreMasterSurf);
    Cur_x2_p = sum(Nb .* CurMasterSurfXYZ_p, 1)';

    % Relative motion and projection
    gs = Cur_x2_p - Cur_x1;
    tv = ContactPairs(i).pc * gs;

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
        ContactPairs(i).CurContactState = 1; % Stick
    else
        ContactPairs(i).CurContactState = 2; % Slip
    end

    % Update stiffness and force
    [GKF, Residual, ContactPairs] = CalculateContactKandF(FEMod, ...
        ContactPairs, Dt, PreDisp, i, GKF, Residual, Disp, IntegralPoint);
end
end