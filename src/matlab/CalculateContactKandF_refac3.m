function [GKF, Residual, ContactPairs] = CalculateContactKandF(FEMod, ContactPairs, Dt, ...
    PreDisp, i, GKF, Residual, Disp, IntegralPoint)
% === Obtain contact stiffness and contact force ===
% This refactor preserves the original algorithm and numeric behaviour.
%
% Note: ContactPairs is a struct-of-vectors: fields are arrays/matrices,
%       use ContactPairs.Field(:, i) for 2xN / matrix fields and
%       ContactPairs.Field(i) for scalar/vector fields.

FricFac = FEMod.FricFac;

if ContactPairs.CurContactState(i) == 1 || ContactPairs.CurContactState(i) == 2
     % --- current slave geometry & previous slave geometry ---
    CurIP = IntegralPoint(ContactPairs.SlaveIntegralPoint(i), :)';
    [Na, N1a, N2a] = GetSurfaceShapeFunction(CurIP(1), CurIP(2));
    [CurSlaveSurfXYZ, SlaveSurfDOF] = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs.SlaveSurf(:, i));
    [PreSlaveSurfNodeXYZ, ~]            = GetSurfaceNodeLocation(FEMod, PreDisp, ContactPairs.SlaveSurf(:, i)); % only needed for slip

    [Cur_n, J1, Cur_N1Xa, Cur_N2Xa, Cur_x1] = getSurfaceGeometry(Na, N1a, N2a, CurSlaveSurfXYZ);
    [~, ~, Pre_N1Xa, Pre_N2Xa, Pre_x1] = getSurfaceGeometry(Na, N1a, N2a, PreSlaveSurfNodeXYZ); % only needed for slip

end

if ContactPairs.CurContactState(i) == 1  % --- Stick contact state ---

    % --- master geometry (previous master surface) ---
    [Nb, ~, ~] = GetSurfaceShapeFunction(ContactPairs.rp(i), ContactPairs.sp(i));
    [CurMasterSurfXYZ_rpsp, MasterSurfDOF] = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs.PreMasterSurf(:, i));
    Cur_x2_p = sum(Nb .* CurMasterSurfXYZ_rpsp, 1)';

    % --- relative sliding vector and traction ---
    gs = Cur_x2_p - Cur_x1;
    tv = ContactPairs.pc(i) * gs;
    ContactNodeForce = AssembleContactForce(Na, Nb, J1, tv);

    ContactDOF = [SlaveSurfDOF; MasterSurfDOF];
    Residual(ContactDOF, :) = Residual(ContactDOF, :) + ContactNodeForce;

    ContactPairs.Pressure(i) = abs(tv' * Cur_n);
    ContactPairs.Traction(i) = abs(sqrt(tv' * tv));             % save pressure/traction

    % --- stiffness components for stick case ---
    Cur_g1_hat_slave = TransVect2SkewSym(Cur_N1Xa);
    Cur_g2_hat_slave = TransVect2SkewSym(Cur_N2Xa);
    Ac = (kron(N1a', Cur_g2_hat_slave) - kron(N2a', Cur_g1_hat_slave)) / J1;
    
    Stick_K = getStickK(Na, Nb, pc, tv, Ac, Cur_n, J1)

    GKF(ContactDOF, ContactDOF) = GKF(ContactDOF, ContactDOF) - Stick_K;

elseif ContactPairs.CurContactState(i) == 2  % --- Slip contact state ---
    tn = ContactPairs.Cur_g(i) * ContactPairs.pc(i);
        
    dx1 = Cur_x1 - Pre_x1;
    PN = eye(3) - Cur_n * Cur_n';

    dg1_slave = Cur_N1Xa - Pre_N1Xa;
    dg2_slave = Cur_N2Xa - Pre_N2Xa;
    m1 = cross(dg1_slave, Cur_N2Xa) + cross(Cur_N1Xa, dg2_slave);
    c1 = PN * m1 / J1;

    % --- master geometry at current and previous steps ---
    [Nb, N1b, N2b] = GetSurfaceShapeFunction(ContactPairs.rc(i), ContactPairs.sc(i));
    [CurMasterSurfNodeXYZ, MasterSurfDOF] = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs.CurMasterSurf(:, i));
    [PreMasterSurfNodeXYZ, ~]             = GetSurfaceNodeLocation(FEMod, PreDisp, ContactPairs.CurMasterSurf(:, i));

    [~, ~, Cur_N1Xb, Cur_N2Xb, Cur_x2] = getSurfaceGeometry(Nb, N1b, N2b, CurMasterSurfNodeXYZ);
    [~, ~, ~, ~, Pre_x2] = getSurfaceGeometry(Nb, N1b, N2b, PreMasterSurfNodeXYZ);

    dx2 = Cur_x2 - Pre_x2;

    % --- relative velocity and tangential direction ---
    r1 = ContactPairs.Cur_g(i) * c1 + dx1 - dx2;
    vr = r1 / Dt;
    s1_temp = PN * vr;

    if sqrt(s1_temp' * s1_temp) > 1e-8
        s1 = s1_temp / sqrt(s1_temp' * s1_temp);
    else
        s1 = zeros(3, 1);
        dh = 0;  %#ok<NASGU>
    end

    % --- contact nodal force (with friction) ---
    tv = tn * (Cur_n + FricFac * s1);
    ContactNodeForce = AssembleContactForce(Na, Nb, J1, tv);

    ContactDOF = [SlaveSurfDOF; MasterSurfDOF];
    Residual(ContactDOF, :) = Residual(ContactDOF, :) + ContactNodeForce;

    ContactPairs.Pressure(i)  = abs(tn);
    ContactPairs.Traction(i) = abs(sqrt(tv' * tv));  % save pressure/traction

    % --- precompute projection matrices and related arrays ---
    A_ab = [ Cur_N1Xa' * Cur_N1Xb, Cur_N1Xa' * Cur_N2Xb;
             Cur_N2Xa' * Cur_N1Xb, Cur_N2Xa' * Cur_N2Xb ];
    a_ab = inv(A_ab);

    g1_bar_slave  = a_ab(1,1) * Cur_N1Xb + a_ab(2,1) * Cur_N2Xb;
    g2_bar_slave  = a_ab(1,2) * Cur_N1Xb + a_ab(2,2) * Cur_N2Xb;
    g1_bar_master = a_ab(1,1) * Cur_N1Xa + a_ab(1,2) * Cur_N2Xa;
    g2_bar_master = a_ab(2,1) * Cur_N1Xa + a_ab(2,2) * Cur_N2Xa;

    N1  = Cur_n * Cur_n';
    N1_bar = eye(3) - Cur_N1Xa * g1_bar_slave' - Cur_N2Xa * g2_bar_slave';

    mc1_bar = kron(N1a', g1_bar_slave) + kron(N2a', g2_bar_slave);
    mb2_bar = kron(N1b', g1_bar_master) + kron(N2b', g2_bar_master);

    Cur_g1_hat_slave = TransVect2SkewSym(Cur_N1Xa);
    Cur_g2_hat_slave = TransVect2SkewSym(Cur_N2Xa);
    Ac = (kron(N1a', Cur_g2_hat_slave) - kron(N2a', Cur_g1_hat_slave)) / J1;

    N1_wave = Cur_n * (N1_bar * Cur_n)';

    % Mc1_bar & Mb2_bar arranged as in original code
    Mc1_bar = [ Cur_n * mc1_bar(:,1)' , Cur_n * mc1_bar(:,2)' , Cur_n * mc1_bar(:,3)' , Cur_n * mc1_bar(:,4)' ];
    Mb2_bar = [ -Cur_n * mb2_bar(:,1)' , -Cur_n * mb2_bar(:,2)' , -Cur_n * mb2_bar(:,3)' , -Cur_n * mb2_bar(:,4)' ];

    N12a = [N1a'; N2a'];
    N12b = [N1b'; N2b'];
    Gbc = ContactPairs.Cur_g(i) * (N12b' * a_ab * N12a);

    FrictionlessK = getFrictionlessK(Na, Nb, ContactPairs.pc(i), tn, Ac, Mc1_bar, Mb2_bar, Gbc, N1, N1_wave, J1);

    Frictional_K = zeros(24);

    % --- frictional additions (only if friction != 0 and s1 nonzero) ---
    if FricFac ~= 0 && sqrt(s1' * s1) > 1e-8
        Q1 = ((Cur_n' * m1) * eye(3) + Cur_n * m1') / J1;
        dg1_hat_slave = TransVect2SkewSym(dg1_slave);
        dg2_hat_slave = TransVect2SkewSym(dg2_slave);
        Ac1_bar = (kron(N1a', dg2_hat_slave) - kron(N2a', dg1_hat_slave)) / J1;

        dh = sqrt((PN * vr)' * (PN * vr)) * Dt;
        Ps = (eye(3) - s1 * s1') / dh;

        R1 = ((Cur_n' * r1) * eye(3) + Cur_n * r1') / ContactPairs.Cur_g(i);
        B1 = (Ps * c1) * (N1_bar * Cur_n)' - Ps * PN;
        L1 = ContactPairs.Cur_g(i) * Ps * (PN * Q1 + R1 - eye(3)) * PN;

        Jc1 = L1 * Ac - ContactPairs.Cur_g(i) * Ps * PN * Ac1_bar;

        hc1_add = N1 * mc1_bar + Ac * kron(eye(4), Cur_n);
        hc1_sub = N1 * mc1_bar - Ac * kron(eye(4), Cur_n);

        S1 = s1 * Cur_n';
        S1_wave = s1 * (N1_bar * Cur_n)';

        Frictional_K11 = zeros(12); Frictional_K12 = zeros(12);
        Frictional_K21 = zeros(12); Frictional_K22 = zeros(12);

        for aa = 1:4
            for bb = 1:4
                idxA = (3*aa-2):(3*aa); idxB = (3*bb-2):(3*bb);

                tempK = ( -Na(aa) * Na(bb) * FricFac * (ContactPairs.pc(i) * S1_wave + tn * B1) ...
                          - Na(aa) * FricFac * tn * ( s1 * hc1_sub(:, bb)' + ContactPairs.Cur_g(i) * Ps * c1 * hc1_add(:, bb)' - Jc1(:, idxB) ) ) * J1;
                Frictional_K11(idxA, idxB) = Frictional_K11(idxA, idxB) + tempK;

                tempK = ( Na(aa) * Nb(bb) * FricFac * (ContactPairs.pc(i) * S1_wave + tn * B1) ) * J1;
                Frictional_K12(idxA, idxB) = Frictional_K12(idxA, idxB) + tempK;

                tempK = ( Nb(aa) * Na(bb) * FricFac * (ContactPairs.pc(i) * S1_wave + tn * B1) ...
                          + Nb(aa) * FricFac * tn * ( s1 * hc1_sub(:, bb)' + ContactPairs.Cur_g(i) * Ps * c1 * hc1_add(:, bb)' - Jc1(:, idxB) ) ...
                          + Na(bb) * FricFac * tn * ( - s1 * mb2_bar(:, aa)' ) + Gbc(aa, bb) * FricFac * tn * S1 ) * J1;
                Frictional_K21(idxA, idxB) = Frictional_K21(idxA, idxB) + tempK;

                tempK = ( -Nb(aa) * Nb(bb) * FricFac * (ContactPairs.pc(i) * S1_wave + tn * B1) ...
                          - Nb(bb) * FricFac * tn * ( - s1 * mb2_bar(:, aa)' ) ) * J1;
                Frictional_K22(idxA, idxB) = Frictional_K22(idxA, idxB) + tempK;
            end
        end

        Frictional_K = [Frictional_K11, Frictional_K12; Frictional_K21, Frictional_K22];
    end

    ContactK = FrictionlessK + Frictional_K;
    GKF(ContactDOF, ContactDOF) = GKF(ContactDOF, ContactDOF) - ContactK;
end

end


function [n, J, N1X, N2X, x] = getSurfaceGeometry(N, N1, N2, SurfXYZ)
x   = sum(N .* SurfXYZ,1)';
N1X = sum(N1 .* SurfXYZ,1)';
N2X = sum(N2 .* SurfXYZ,1)';
n = cross(N1X, N2X);
n = n / norm(n);
J = norm(cross(N1X, N2X));
end

function ContactNodeForce = AssembleContactForce(Na, Nb, J1, tv)
ContactNodeForce = zeros(24,1);
for a = 1:4
    ContactNodeForce((3*a-2):3*a)       = Na(a) * J1 * tv;
    ContactNodeForce(((3*a-2):3*a)+12) = -Nb(a) * J1 * tv;
end
end

function Stick_K = getStickK(Na, Nb, pc, tv, Ac, Cur_n, J1)
Stick_K11 = zeros(12); Stick_K12 = zeros(12);
Stick_K21 = zeros(12); Stick_K22 = zeros(12);
for aa = 1:4
    for bb = 1:4
        idxA = (3*aa-2):(3*aa); idxB = (3*bb-2):(3*bb);
        tempK = (-Na(aa)*Na(bb)*pc*eye(3) - Na(aa)*(-tv*(Ac(:,idxB)*Cur_n)'))*J1;
        Stick_K11(idxA, idxB) = Stick_K11(idxA, idxB) + tempK;
        tempK = (Na(aa)*Nb(bb)*pc*eye(3))*J1; Stick_K12(idxA, idxB)=Stick_K12(idxA, idxB)+tempK;
        tempK = (Nb(aa)*Na(bb)*pc*eye(3)+Nb(aa)*(-tv*(Ac(:,idxB)*Cur_n)'))*J1; Stick_K21(idxA, idxB)=Stick_K21(idxA, idxB)+tempK;
        tempK = (-Nb(aa)*Nb(bb)*pc*eye(3))*J1; Stick_K22(idxA, idxB)=Stick_K22(idxA, idxB)+tempK;
    end
end
Stick_K = [Stick_K11, Stick_K12; Stick_K21, Stick_K22];
end

function FrictionlessK = getFrictionlessK(Na, Nb, pc, tn, Ac, Mc1_bar, Mb2_bar, Gbc, N1, N1_wave, J1)
Frictionless_K11 = zeros(12); Frictionless_K12 = zeros(12);
Frictionless_K21 = zeros(12); Frictionless_K22 = zeros(12);

for aa = 1:4
    for bb = 1:4
        idxA = (3*aa-2):(3*aa); idxB = (3*bb-2):(3*bb);
        tempK = (-Na(aa)*Na(bb)*pc*N1_wave - Na(aa)*tn*(Ac(:,idxB)+Mc1_bar(:,idxB)*N1))*J1;
        Frictionless_K11(idxA, idxB) = Frictionless_K11(idxA, idxB) + tempK;
        tempK = (Na(aa)*Nb(bb)*pc*N1_wave)*J1;
        Frictionless_K12(idxA, idxB) = Frictionless_K12(idxA, idxB) + tempK;
        tempK = (Nb(aa)*Na(bb)*pc*N1_wave + Nb(aa)*tn*(Ac(:,idxB)+Mc1_bar(:,idxB)*N1) + Na(bb)*tn*Mb2_bar(:,idxA) + Gbc(aa,bb)*tn*N1)*J1;
        Frictionless_K21(idxA, idxB) = Frictionless_K21(idxA, idxB) + tempK;
        tempK = (-Nb(aa)*Nb(bb)*pc*N1_wave - Nb(bb)*tn*Mb2_bar(:,idxA))*J1;
        Frictionless_K22(idxA, idxB) = Frictionless_K22(idxA, idxB) + tempK;
    end
end
FrictionlessK = [Frictionless_K11, Frictionless_K12; Frictionless_K21, Frictionless_K22];
end