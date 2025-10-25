function [GKF, Residual, ContactPairs] = CalculateContactKandF_stick(FEMod, ContactPairs, Dt, ...
    PreDisp, i, GKF, Residual, Disp, IntegralPoint)

FricFac = FEMod.FricFac;

% --- slave geometry at current IP ---
CurIP = IntegralPoint(ContactPairs.SlaveIntegralPoint(i), :)';
[Na, N1a, N2a] = GetSurfaceShapeFunction(CurIP(1), CurIP(2));
[CurSlaveSurfXYZ, SlaveSurfDOF] = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs.SlaveSurf(:, i));

Cur_x1    = sum(Na .* CurSlaveSurfXYZ, 1)';                 % slave contact point
Cur_N1Xa  = sum(N1a .* CurSlaveSurfXYZ, 1)';                % slave tangent 1
Cur_N2Xa  = sum(N2a .* CurSlaveSurfXYZ, 1)';                % slave tangent 2

Cur_n = cross(Cur_N1Xa, Cur_N2Xa);
Cur_n = Cur_n / norm(Cur_n);
J1 = norm(cross(Cur_N1Xa, Cur_N2Xa));                       % surface Jacobian

% --- master geometry (previous master surface) ---
[Nb, ~, ~] = GetSurfaceShapeFunction(ContactPairs.rp(i), ContactPairs.sp(i));
[CurMasterSurfXYZ_rpsp, MasterSurfDOF] = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs.PreMasterSurf(:, i));
Cur_x2_p = sum(Nb .* CurMasterSurfXYZ_rpsp, 1)';

% --- relative sliding vector and traction ---
gs = Cur_x2_p - Cur_x1;
tv = ContactPairs.pc(i) * gs;

ContactPairs.Pressure(i) = abs(tv' * Cur_n);
ContactPairs.Traction(i) = abs(sqrt(tv' * tv));             % save pressure/traction

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
        tempK = ( -Na(aa) * Na(bb) * ContactPairs.pc(i) * eye(3) ...
                  - Na(aa) * ( -tv * (Ac(:, (3*bb-2):(3*bb)) * Cur_n)' ) ) * J1;
        idxA = (3*aa-2):(3*aa); idxB = (3*bb-2):(3*bb);
        Stick_K11(idxA, idxB) = Stick_K11(idxA, idxB) + tempK;

        % K12 term
        tempK = ( Na(aa) * Nb(bb) * ContactPairs.pc(i) * eye(3) ) * J1;
        Stick_K12(idxA, idxB) = Stick_K12(idxA, idxB) + tempK;

        % K21 term
        tempK = ( Nb(aa) * Na(bb) * ContactPairs.pc(i) * eye(3) ...
                  + Nb(aa) * ( -tv * (Ac(:, (3*bb-2):(3*bb)) * Cur_n)' ) ) * J1;
        Stick_K21(idxA, idxB) = Stick_K21(idxA, idxB) + tempK;

        % K22 term
        tempK = ( -Nb(aa) * Nb(bb) * ContactPairs.pc(i) * eye(3) ) * J1;
        Stick_K22(idxA, idxB) = Stick_K22(idxA, idxB) + tempK;
    end
end

Stick_K = [Stick_K11, Stick_K12; Stick_K21, Stick_K22];
GKF(ContactDOF, ContactDOF) = GKF(ContactDOF, ContactDOF) - Stick_K;
end