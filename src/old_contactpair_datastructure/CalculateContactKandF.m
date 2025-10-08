function [GKF, Residual, ContactPairs] = CalculateContactKandF(FEMod, ContactPairs, Dt, ...
    PreDisp, i, GKF, Residual, Disp, IntegralPoint)
% === Obtain contact stiffness and contact force ===
% This refactor preserves the original algorithm and numeric behaviour.

FricFac = FEMod.FricFac;

if ContactPairs(i).CurContactState == 1  % --- Stick contact state ---
    % --- slave geometry at current IP ---
    CurIP = IntegralPoint(ContactPairs(i).SlaveIntegralPoint, :)';
    [Na, N1a, N2a] = GetSurfaceShapeFunction(CurIP(1), CurIP(2));
    [CurSlaveSurfXYZ, SlaveSurfDOF] = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs(i).SlaveSurf);

    Cur_x1    = sum(Na .* CurSlaveSurfXYZ, 1)';                 % slave contact point
    Cur_N1Xa  = sum(N1a .* CurSlaveSurfXYZ, 1)';                % slave tangent 1
    Cur_N2Xa  = sum(N2a .* CurSlaveSurfXYZ, 1)';                % slave tangent 2

    Cur_n = cross(Cur_N1Xa, Cur_N2Xa);
    Cur_n = Cur_n / norm(Cur_n);
    J1 = norm(cross(Cur_N1Xa, Cur_N2Xa));                       % surface Jacobian

    % --- master geometry (previous master surface) ---
    [Nb, ~, ~] = GetSurfaceShapeFunction(ContactPairs(i).rp, ContactPairs(i).sp);
    [CurMasterSurfXYZ_rpsp, MasterSurfDOF] = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs(i).PreMasterSurf);
    Cur_x2_p = sum(Nb .* CurMasterSurfXYZ_rpsp, 1)';

    % --- relative sliding vector and traction ---
    gs = Cur_x2_p - Cur_x1;
    tv = ContactPairs(i).pc * gs;

    ContactPairs(i).Pressure = abs(tv' * Cur_n);
    ContactPairs(i).Traction = abs(sqrt(tv' * tv));             % save pressure/traction

    % --- Assemble contact nodal force (24 x 1) ---
    ContactNodeForce = zeros(24, 1);
    ContactDOF = [SlaveSurfDOF; MasterSurfDOF];
    for a = 1:4
        ContactNodeForce((3*a-2):3*a)       = Na(a) * J1 * tv;
        ContactNodeForce(((3*a-2):3*a) + 12) = -Nb(a) * J1 * tv;
    end
    Residual(ContactDOF, :) = Residual(ContactDOF, :) + ContactNodeForce;

    % --- stiffness components for stick case ---
    Cur_g1_hat_slave = TransVect2SkewSym(Cur_N1Xa);
    Cur_g2_hat_slave = TransVect2SkewSym(Cur_N2Xa);
    Ac = (kron(N1a', Cur_g2_hat_slave) - kron(N2a', Cur_g1_hat_slave)) / J1;

    Stick_K11 = zeros(12); Stick_K12 = zeros(12);
    Stick_K21 = zeros(12); Stick_K22 = zeros(12);

    for aa = 1:4
        for bb = 1:4
            % K11 term
            tempK = ( -Na(aa) * Na(bb) * ContactPairs(i).pc * eye(3) ...
                      - Na(aa) * ( -tv * (Ac(:, (3*bb-2):(3*bb)) * Cur_n)' ) ) * J1;
            idxA = (3*aa-2):(3*aa); idxB = (3*bb-2):(3*bb);
            Stick_K11(idxA, idxB) = Stick_K11(idxA, idxB) + tempK;

            % K12 term
            tempK = ( Na(aa) * Nb(bb) * ContactPairs(i).pc * eye(3) ) * J1;
            Stick_K12(idxA, idxB) = Stick_K12(idxA, idxB) + tempK;

            % K21 term
            tempK = ( Nb(aa) * Na(bb) * ContactPairs(i).pc * eye(3) ...
                      + Nb(aa) * ( -tv * (Ac(:, (3*bb-2):(3*bb)) * Cur_n)' ) ) * J1;
            Stick_K21(idxA, idxB) = Stick_K21(idxA, idxB) + tempK;

            % K22 term
            tempK = ( -Nb(aa) * Nb(bb) * ContactPairs(i).pc * eye(3) ) * J1;
            Stick_K22(idxA, idxB) = Stick_K22(idxA, idxB) + tempK;
        end
    end

    Stick_K = [Stick_K11, Stick_K12; Stick_K21, Stick_K22];
    GKF(ContactDOF, ContactDOF) = GKF(ContactDOF, ContactDOF) - Stick_K;

elseif ContactPairs(i).CurContactState == 2  % --- Slip contact state ---
    tn = ContactPairs(i).Cur_g * ContactPairs(i).pc;

    % --- current slave geometry & previous slave geometry ---
    CurIP = IntegralPoint(ContactPairs(i).SlaveIntegralPoint, :)';
    [Na, N1a, N2a] = GetSurfaceShapeFunction(CurIP(1), CurIP(2));
    [CurSlaveSurfXYZ, SlaveSurfDOF]     = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs(i).SlaveSurf);
    [PreSlaveSurfNodeXYZ, ~]            = GetSurfaceNodeLocation(FEMod, PreDisp, ContactPairs(i).SlaveSurf);

    Cur_x1 = sum(Na .* CurSlaveSurfXYZ, 1)';  Pre_x1 = sum(Na .* PreSlaveSurfNodeXYZ, 1)';
    dx1 = Cur_x1 - Pre_x1;

    Pre_N1Xa = sum(N1a .* PreSlaveSurfNodeXYZ, 1)'; Pre_N2Xa = sum(N2a .* PreSlaveSurfNodeXYZ, 1)';
    Cur_N1Xa = sum(N1a .* CurSlaveSurfXYZ, 1)';   Cur_N2Xa = sum(N2a .* CurSlaveSurfXYZ, 1)';

    Cur_n = cross(Cur_N1Xa, Cur_N2Xa); Cur_n = Cur_n / norm(Cur_n);
    J1 = norm(cross(Cur_N1Xa, Cur_N2Xa));
    PN = eye(3) - Cur_n * Cur_n';

    dg1_slave = Cur_N1Xa - Pre_N1Xa;
    dg2_slave = Cur_N2Xa - Pre_N2Xa;
    m1 = cross(dg1_slave, Cur_N2Xa) + cross(Cur_N1Xa, dg2_slave);
    c1 = PN * m1 / J1;

    % --- master geometry at current and previous steps ---
    [Nb, N1b, N2b] = GetSurfaceShapeFunction(ContactPairs(i).rc, ContactPairs(i).sc);
    [CurMasterSurfNodeXYZ, MasterSurfDOF] = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs(i).CurMasterSurf);
    [PreMasterSurfNodeXYZ, ~]             = GetSurfaceNodeLocation(FEMod, PreDisp, ContactPairs(i).CurMasterSurf);

    Cur_x2 = sum(Nb .* CurMasterSurfNodeXYZ, 1)';   Pre_x2 = sum(Nb .* PreMasterSurfNodeXYZ, 1)';
    Cur_N1Xb = sum(N1b .* CurMasterSurfNodeXYZ, 1)'; Cur_N2Xb = sum(N2b .* CurMasterSurfNodeXYZ, 1)';
    dx2 = Cur_x2 - Pre_x2;

    % --- relative velocity and tangential direction ---
    r1 = ContactPairs(i).Cur_g * c1 + dx1 - dx2;
    vr = r1 / Dt;
    s1_temp = PN * vr;

    if sqrt(s1_temp' * s1_temp) > 1e-8
        s1 = s1_temp / sqrt(s1_temp' * s1_temp);
    else
        s1 = zeros(3, 1);
        dh = 0;  %#ok<NASGU>
    end

    % --- contact nodal force (with friction) ---
    ContactNodeForce = zeros(24, 1);
    tv = tn * (Cur_n + FricFac * s1);
    temp_f1a = tv * J1;

    ContactPairs(i).Pressure  = abs(tn);
    ContactPairs(i).Traction = abs(sqrt(tv' * tv));  % save pressure/traction

    for a = 1:4
        f1a = Na(a) * temp_f1a;
        f2b = -Nb(a) * temp_f1a;
        ContactNodeForce((3*a-2):3*a)       = f1a;
        ContactNodeForce(((3*a-2):3*a) + 12) = f2b;
    end

    ContactDOF = [SlaveSurfDOF; MasterSurfDOF];
    Residual(ContactDOF, :) = Residual(ContactDOF, :) + ContactNodeForce;

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
    Gbc = ContactPairs(i).Cur_g * (N12b' * a_ab * N12a);

    % --- frictionless stiffness baseline ---
    Frictionless_K11 = zeros(12); Frictionless_K12 = zeros(12);
    Frictionless_K21 = zeros(12); Frictionless_K22 = zeros(12);

    for aa = 1:4
        for bb = 1:4
            idxA = (3*aa-2):(3*aa); idxB = (3*bb-2):(3*bb);

            tempK = ( -Na(aa) * Na(bb) * ContactPairs(i).pc * N1_wave ...
                      - Na(aa) * tn * (Ac(:, idxB) + Mc1_bar(:, idxB) * N1) ) * J1;
            Frictionless_K11(idxA, idxB) = Frictionless_K11(idxA, idxB) + tempK;

            tempK = ( Na(aa) * Nb(bb) * ContactPairs(i).pc * N1_wave ) * J1;
            Frictionless_K12(idxA, idxB) = Frictionless_K12(idxA, idxB) + tempK;

            tempK = ( Nb(aa) * Na(bb) * ContactPairs(i).pc * N1_wave ...
                      + Nb(aa) * tn * (Ac(:, idxB) + Mc1_bar(:, idxB) * N1) ...
                      + Na(bb) * tn * Mb2_bar(:, idxA) + Gbc(aa, bb) * tn * N1 ) * J1;
            Frictionless_K21(idxA, idxB) = Frictionless_K21(idxA, idxB) + tempK;

            tempK = ( -Nb(aa) * Nb(bb) * ContactPairs(i).pc * N1_wave ...
                      - Nb(bb) * tn * Mb2_bar(:, idxA) ) * J1;
            Frictionless_K22(idxA, idxB) = Frictionless_K22(idxA, idxB) + tempK;
        end
    end

    FrictionlessK = [Frictionless_K11, Frictionless_K12; Frictionless_K21, Frictionless_K22];

    Frictional_K = zeros(24);

    % --- frictional additions (only if friction != 0 and s1 nonzero) ---
    if FricFac ~= 0 && sqrt(s1' * s1) > 1e-8
        Q1 = ((Cur_n' * m1) * eye(3) + Cur_n * m1') / J1;
        dg1_hat_slave = TransVect2SkewSym(dg1_slave);
        dg2_hat_slave = TransVect2SkewSym(dg2_slave);
        Ac1_bar = (kron(N1a', dg2_hat_slave) - kron(N2a', dg1_hat_slave)) / J1;

        dh = sqrt((PN * vr)' * (PN * vr)) * Dt;
        Ps = (eye(3) - s1 * s1') / dh;

        R1 = ((Cur_n' * r1) * eye(3) + Cur_n * r1') / ContactPairs(i).Cur_g;
        B1 = (Ps * c1) * (N1_bar * Cur_n)' - Ps * PN;
        L1 = ContactPairs(i).Cur_g * Ps * (PN * Q1 + R1 - eye(3)) * PN;

        Jc1 = L1 * Ac - ContactPairs(i).Cur_g * Ps * PN * Ac1_bar;

        hc1_add = N1 * mc1_bar + Ac * kron(eye(4), Cur_n);
        hc1_sub = N1 * mc1_bar - Ac * kron(eye(4), Cur_n);

        S1 = s1 * Cur_n';
        S1_wave = s1 * (N1_bar * Cur_n)';

        Frictional_K11 = zeros(12); Frictional_K12 = zeros(12);
        Frictional_K21 = zeros(12); Frictional_K22 = zeros(12);

        for aa = 1:4
            for bb = 1:4
                idxA = (3*aa-2):(3*aa); idxB = (3*bb-2):(3*bb);

                tempK = ( -Na(aa) * Na(bb) * FricFac * (ContactPairs(i).pc * S1_wave + tn * B1) ...
                          - Na(aa) * FricFac * tn * ( s1 * hc1_sub(:, bb)' + ContactPairs(i).Cur_g * Ps * c1 * hc1_add(:, bb)' - Jc1(:, idxB) ) ) * J1;
                Frictional_K11(idxA, idxB) = Frictional_K11(idxA, idxB) + tempK;

                tempK = ( Na(aa) * Nb(bb) * FricFac * (ContactPairs(i).pc * S1_wave + tn * B1) ) * J1;
                Frictional_K12(idxA, idxB) = Frictional_K12(idxA, idxB) + tempK;

                tempK = ( Nb(aa) * Na(bb) * FricFac * (ContactPairs(i).pc * S1_wave + tn * B1) ...
                          + Nb(aa) * FricFac * tn * ( s1 * hc1_sub(:, bb)' + ContactPairs(i).Cur_g * Ps * c1 * hc1_add(:, bb)' - Jc1(:, idxB) ) ...
                          + Na(bb) * FricFac * tn * ( - s1 * mb2_bar(:, aa)' ) + Gbc(aa, bb) * FricFac * tn * S1 ) * J1;
                Frictional_K21(idxA, idxB) = Frictional_K21(idxA, idxB) + tempK;

                tempK = ( -Nb(aa) * Nb(bb) * FricFac * (ContactPairs(i).pc * S1_wave + tn * B1) ...
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