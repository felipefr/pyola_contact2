function ContactPairs = ContactSearch(FEMod, ContactPairs, Disp, IntegralPoint)
% === Find nearest master surface corresponding to slave surface point ===

for i = 1:size(ContactPairs, 1)

    % --- Get current slave surface geometry ---
    [SlaveSurfNodeXYZ, ~] = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs(i).SlaveSurf);

    % Current integration point coordinates
    CurIP = IntegralPoint(ContactPairs(i).SlaveIntegralPoint, :)';
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
        ContactPairs(i).CurMasterSurf = [MasterEle; MasterSign];
        ContactPairs(i).rc   = rr;
        ContactPairs(i).sc   = ss;
        ContactPairs(i).Cur_g = gg;
    else
        ContactPairs(i).CurMasterSurf = [0; 0];
        ContactPairs(i).rc   = 0;
        ContactPairs(i).sc   = 0;
        ContactPairs(i).Cur_g = 0;
        ContactPairs(i).CurContactState = 0;
    end
end
end