% It is weird, but this function lead to faster convergence. 
function [ContactPairs, GKF, Residual] = DetermineContactState_chatgpt(FEMod, ContactPairs, ...
    Dt, PreDisp, GKF, Residual, Disp)

% === Determine contact state, then calculate contact stiffness and contact force ===

% 2×2 Gauss points
gp = 1/sqrt(3);
IntegralPoint = [-gp, -gp; gp, -gp; gp, gp; -gp, gp];
FricFac = FEMod.FricFac;

% --- Step 1: Search for current contact pairs
ContactPairs = ContactSearch(FEMod, ContactPairs, Disp, IntegralPoint);

% --- Step 2: Loop over possible contact pairs
for i = 1:numel(ContactPairs)

    pair = ContactPairs(i);

    % Skip if no current master surface (no contact)
    if pair.CurMasterSurf(1) == 0
        continue;
    end

    % --- Step 3: Determine current contact state ---
    if FricFac == 0 || pair.PreMasterSurf(1) == 0
        % First contact or frictionless → slip
        pair.CurContactState = 2;

    else
        % Retrieve current slave surface geometry
        ip = IntegralPoint(pair.SlaveIntegralPoint, :).';
        [Na, N1a, N2a] = GetSurfaceShapeFunction(ip(1), ip(2));
        CurSlaveSurfXYZ = GetSurfaceNodeLocation(FEMod, Disp, pair.SlaveSurf);
        Cur_x1 = (Na * CurSlaveSurfXYZ).';

        % Retrieve previous master surface geometry
        [Nb, ~, ~] = GetSurfaceShapeFunction(pair.rp, pair.sp);
        CurMasterSurfXYZ_p = GetSurfaceNodeLocation(FEMod, Disp, pair.PreMasterSurf);
        Cur_x2_p = (Nb * CurMasterSurfXYZ_p).';

        % Relative tangential displacement
        gs = Cur_x2_p - Cur_x1;
        tv = pair.pc * gs;

        % Current normal vector
        Cur_N1Xa = (N1a * CurSlaveSurfXYZ).';
        Cur_N2Xa = (N2a * CurSlaveSurfXYZ).';
        Cur_n = cross(Cur_N1Xa, Cur_N2Xa);
        Cur_n = Cur_n / norm(Cur_n);

        % Trial tangential and normal traction components
        tn = abs(tv' * Cur_n);
        tt = sqrt(norm(tv)^2 - tn^2);

        % Slip/stick criterion
        pair.CurContactState = 1 + (tt - FricFac * tn >= 0);
    end

    % --- Step 4: Assemble contact contribution ---
    [GKF, Residual, ContactPairs] = CalculateContactKandF(FEMod, ...
        ContactPairs, Dt, PreDisp, i, GKF, Residual, Disp, IntegralPoint);
end
end