function [rr, ss, MasterEle, MasterSign, gg, Exist] = GetContactPointbyRayTracing(FEMod, Disp, SlavePoint, SlavePointTan)
% === Obtain master surface contact point by RayTracing ===

Tol = 1e-4;
Exist = -1;
MinDis = 1e8;
MinGrow = 0;
Ming = -1e3;
MinMasterPoint = [];

AllMasterNode = zeros(size(FEMod.MasterSurf, 2), 4);

% --- Find node closest to integration point from slave surface ---
for i = 1:size(FEMod.MasterSurf, 2)
    MasterSurfNode = GetSurfaceNode(FEMod.Eles(FEMod.MasterSurf(1, i), :), FEMod.MasterSurf(2, i));
    AllMasterNode(i, :) = MasterSurfNode;

    % Build DOF list for master surface nodes
    MasterSurfDOF = zeros(numel(MasterSurfNode) * 3, 1);
    for m = 1:numel(MasterSurfNode)
        MasterSurfDOF((3*m - 2):(3*m)) = (3*MasterSurfNode(m) - 2):(3*MasterSurfNode(m));
    end

    % Compute current deformed coordinates
    MasterSurfDis = reshape(Disp(MasterSurfDOF), 3, numel(MasterSurfNode))';
    MasterSurfXYZ = FEMod.Nodes(MasterSurfNode, :) + MasterSurfDis;

    % Find nearest node to slave point
    for j = 1:4
        ll = MasterSurfXYZ(j, :)' - SlavePoint;
        Distance = norm(ll);
        if Distance < MinDis
            MinDis = Distance;
            MinMasterPoint = MasterSurfNode(j);
        end
    end
end

% --- Determine master surfaces corresponding to nearest node ---
[AllMinMasterSurfNum, ~] = find(AllMasterNode == MinMasterPoint);
ContactCandidate = zeros(numel(AllMinMasterSurfNum), 8);
ContactCandidate(:, 5) = 1e7;

% --- Loop over candidate master surfaces ---
for i = 1:numel(AllMinMasterSurfNum)

    MasterSurfNode = AllMasterNode(AllMinMasterSurfNum(i), :);

    % Build DOFs
    MasterSurfDOF = zeros(numel(MasterSurfNode) * 3, 1);
    for m = 1:numel(MasterSurfNode)
        MasterSurfDOF((3*m - 2):(3*m)) = (3*MasterSurfNode(m) - 2):(3*MasterSurfNode(m));
    end

    % Deformed coordinates
    MasterSurfDis = reshape(Disp(MasterSurfDOF), 3, numel(MasterSurfNode))';
    MasterSurfXYZ = FEMod.Nodes(MasterSurfNode, :) + MasterSurfDis;

    % Initialize RayTracing parameters
    r = 0; s = 0;

    % --- Newton–Raphson iteration to find (r,s) ---
    for j = 1:1e8
        [N, N1, N2] = GetSurfaceShapeFunction(r, s);

        NX  = sum(N  .* MasterSurfXYZ, 1)';
        N1X = sum(N1 .* MasterSurfXYZ, 1)';
        N2X = sum(N2 .* MasterSurfXYZ, 1)';

        fai = [ (SlavePoint - NX)' * SlavePointTan(:, 1);
                (SlavePoint - NX)' * SlavePointTan(:, 2) ];

        % Abort if NR fails to converge
        if j == 500
            r = 1e5; 
            Exist = -1;
            break;
        end

        if max(abs(fai)) < Tol
            break;
        end

        k11 = N1X' * SlavePointTan(:, 1);
        k12 = N2X' * SlavePointTan(:, 1);
        k21 = N1X' * SlavePointTan(:, 2);
        k22 = N2X' * SlavePointTan(:, 2);

        KT = [k11 k12; k21 k22];
        drs = KT \ fai;

        r = r + drs(1);
        s = s + drs(2);
    end

    % --- Save nearest RayTracing point parameter coordinates ---
    if abs(r) <= 1.01 && abs(s) < 1.01
        v = cross(SlavePointTan(:, 1), SlavePointTan(:, 2));
        v = v / norm(v);

        g = (NX - SlavePoint)' * v;

        ContactCandidate(i, 1) = FEMod.MasterSurf(1, AllMinMasterSurfNum(i));
        ContactCandidate(i, 2) = FEMod.MasterSurf(2, AllMinMasterSurfNum(i));
        ContactCandidate(i, 3:5) = [r, s, g];
        ContactCandidate(i, 6:8) = v';

        % Determine existence & contact type
        if Exist <= 0
            if g >= 0 && abs(Ming) > abs(g)
                Exist = 0; MinGrow = i; Ming = g;
            elseif g < 0
                Exist = 1; MinGrow = i; Ming = g;
            end
        elseif Exist == 1
            if g < 0 && abs(Ming) > abs(g)
                Exist = 1; MinGrow = i; Ming = g;
            end
        end
    end
end

% --- Final contact status and outputs ---
if Exist == 0  % Master surface exists, no contact
    MasterEle  = ContactCandidate(MinGrow, 1);
    MasterSign = ContactCandidate(MinGrow, 2);
    rr = ContactCandidate(MinGrow, 3);
    ss = ContactCandidate(MinGrow, 4);
    gg = ContactCandidate(MinGrow, 5);

elseif Exist == 1  % Contact occurred
    MasterEle  = ContactCandidate(MinGrow, 1);
    MasterSign = ContactCandidate(MinGrow, 2);
    rr = ContactCandidate(MinGrow, 3);
    ss = ContactCandidate(MinGrow, 4);
    gg = ContactCandidate(MinGrow, 5);

else  % No master surface found
    MasterEle = 1e10; MasterSign = 1e10;
    rr = 1e10; ss = 1e10; gg = 1e10;
end

end