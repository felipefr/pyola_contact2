function ContactPairs = ContactSearch(FEMod, ContactPairs, Disp, IntegralPoint)
% === Find nearest master surface corresponding to slave surface point ===

nPairs = size(ContactPairs.SlaveSurf, 2);

for i = 1:nPairs

    % --- Get current slave surface geometry ---
    [SlaveSurfNodeXYZ, ~] = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs.SlaveSurf(:, i));

    % Current integration point coordinates
    CurIP = IntegralPoint(ContactPairs.SlaveIntegralPoint(i), :)';
    [N, N1, N2] = GetSurfaceShapeFunction(CurIP(1), CurIP(2));

    % Compute slave surface point and tangents
    SlavePoint   = sum(N  .* SlaveSurfNodeXYZ, 1)';
    N1X          = sum(N1 .* SlaveSurfNodeXYZ, 1)';
    N2X          = sum(N2 .* SlaveSurfNodeXYZ, 1)';
    SlavePointTan = [N1X, N2X];

    % --- Find nearest master surface via ray tracing ---
    [rr, ss, MasterEle, MasterSign, gg, Exist] = GetContactPointbyRayTracing( ...
        FEMod, Disp, SlavePoint, SlavePointTan);

    % --- Update contact pair information ---
    if Exist == 1
        ContactPairs.CurMasterSurf(:, i) = [MasterEle; MasterSign];
        ContactPairs.rc(i)   = rr;
        ContactPairs.sc(i)   = ss;
        ContactPairs.Cur_g(i) = gg;
    else
        ContactPairs.CurMasterSurf(:, i) = [0; 0];
        ContactPairs.rc(i)   = 0;
        ContactPairs.sc(i)   = 0;
        ContactPairs.Cur_g(i) = 0;
        ContactPairs.CurContactState(i) = 0;
    end
end
end
