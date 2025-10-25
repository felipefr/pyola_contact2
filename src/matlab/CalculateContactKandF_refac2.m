function [GKF, Residual, ContactPairs] = CalculateContactKandF_refac2(FEMod, ContactPairs, Dt, PreDisp, i, GKF, Residual, Disp, IntegralPoint)
% CALCULATECONTACTKANDF Compute contact stiffness and force
% This is the refactored version: split into stick/slip, friction, and helper functions.

state = ContactPairs.CurContactState(i);

switch state
    case 1
        [GKF, Residual, ContactPairs] = AssembleStickContact(FEMod, ContactPairs, i, GKF, Residual, Disp, IntegralPoint);

    case 2
        [GKF, Residual, ContactPairs] = AssembleSlipContact(FEMod, ContactPairs, Dt, i, GKF, Residual, Disp, PreDisp, IntegralPoint);

    otherwise
        % No contact: do nothing
end

end


function [GKF, Residual, ContactPairs] = AssembleStickContact(FEMod, ContactPairs, i, GKF, Residual, Disp, IntegralPoint)
FricFac = FEMod.FricFac;

% --- Slave geometry ---
CurIP = IntegralPoint(ContactPairs.SlaveIntegralPoint(i), :)';
[Na, N1a, N2a] = GetSurfaceShapeFunction(CurIP(1), CurIP(2));
[CurSlaveSurfXYZ, SlaveSurfDOF] = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs.SlaveSurf(:, i));
[Cur_n, J1, Cur_N1Xa, Cur_N2Xa, Cur_x1] = ComputeSlaveSurfaceGeometry(Na, N1a, N2a, CurSlaveSurfXYZ);

% --- Master geometry ---
[Nb, ~, ~] = GetSurfaceShapeFunction(ContactPairs.rp(i), ContactPairs.sp(i));
[CurMasterSurfXYZ_rpsp, MasterSurfDOF] = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs.PreMasterSurf(:, i));
Cur_x2_p = sum(Nb .* CurMasterSurfXYZ_rpsp, 1)';

% --- Relative sliding and traction ---
gs = Cur_x2_p - Cur_x1;
tv = ContactPairs.pc(i) * gs;
ContactPairs.Pressure(i) = abs(tv' * Cur_n);
ContactPairs.Traction(i) = norm(tv);

% --- Assemble contact nodal force ---
ContactDOF = [SlaveSurfDOF; MasterSurfDOF];
ContactNodeForce = AssembleContactForce(Na, Nb, J1, tv);
Residual(ContactDOF, :) = Residual(ContactDOF, :) + ContactNodeForce;

% --- Stiffness ---
Ac = (kron(N1a', TransVect2SkewSym(Cur_N2Xa)) - kron(N2a', TransVect2SkewSym(Cur_N1Xa))) / J1;
Stick_K = AssembleStickStiffness(Na, Nb, ContactPairs.pc(i), tv, Ac, Cur_n, J1);
GKF(ContactDOF, ContactDOF) = GKF(ContactDOF, ContactDOF) - Stick_K;

end

function [GKF, Residual, ContactPairs] = AssembleSlipContact(FEMod, ContactPairs, Dt, i, GKF, Residual, Disp, PreDisp, IntegralPoint)
FricFac = FEMod.FricFac;
tn = ContactPairs.Cur_g(i) * ContactPairs.pc(i);

% --- Slave geometry ---
CurIP = IntegralPoint(ContactPairs.SlaveIntegralPoint(i), :)';
[Na, N1a, N2a] = GetSurfaceShapeFunction(CurIP(1), CurIP(2));
[CurSlaveSurfXYZ, SlaveSurfDOF] = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs.SlaveSurf(:, i));
[PreSlaveSurfNodeXYZ, ~] = GetSurfaceNodeLocation(FEMod, PreDisp, ContactPairs.SlaveSurf(:, i));
[Cur_n, J1, Cur_N1Xa, Cur_N2Xa, Cur_x1] = ComputeSlaveSurfaceGeometry(Na, N1a, N2a, CurSlaveSurfXYZ);
Pre_x1 = sum(Na .* PreSlaveSurfNodeXYZ,1)'; dx1 = Cur_x1 - Pre_x1;
Pre_N1Xa = sum(N1a .* PreSlaveSurfNodeXYZ,1)'; Pre_N2Xa = sum(N2a .* PreSlaveSurfNodeXYZ,1)';

dg1_slave = Cur_N1Xa - Pre_N1Xa;
dg2_slave = Cur_N2Xa - Pre_N2Xa;
PN = eye(3) - Cur_n * Cur_n';
m1 = cross(dg1_slave, Cur_N2Xa) + cross(Cur_N1Xa, dg2_slave);
c1 = PN * m1 / J1;

% --- Master geometry ---
[Nb, N1b, N2b] = GetSurfaceShapeFunction(ContactPairs.rc(i), ContactPairs.sc(i));
[CurMasterSurfNodeXYZ, MasterSurfDOF] = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs.CurMasterSurf(:, i));
[PreMasterSurfNodeXYZ, ~] = GetSurfaceNodeLocation(FEMod, PreDisp, ContactPairs.CurMasterSurf(:, i));
Cur_x2 = sum(Nb .* CurMasterSurfNodeXYZ,1)'; Pre_x2 = sum(Nb .* PreMasterSurfNodeXYZ,1)'; dx2 = Cur_x2 - Pre_x2;
Cur_N1Xb = sum(N1b .* CurMasterSurfNodeXYZ,1)'; Cur_N2Xb = sum(N2b .* CurMasterSurfNodeXYZ,1)';

% --- Relative velocity and tangential direction ---
r1 = ContactPairs.Cur_g(i) * c1 + dx1 - dx2;
vr = r1 / Dt;
s1_temp = PN * vr;
if norm(s1_temp) > 1e-8
    s1 = s1_temp / norm(s1_temp);
else
    s1 = zeros(3,1);
end

% --- Contact force ---
ContactDOF = [SlaveSurfDOF; MasterSurfDOF];
tv = tn * (Cur_n + FricFac * s1);
ContactNodeForce = AssembleContactForce(Na, Nb, J1, tv);
Residual(ContactDOF, :) = Residual(ContactDOF, :) + ContactNodeForce;

% --- Stiffness ---
[ContactPairs] = AssembleSlipK(FEMod, ContactPairs, i, Na, N1a, N2a, Nb, N1b, N2b, Cur_n, Cur_N1Xa, Cur_N2Xa, Cur_N1Xb, Cur_N2Xb, ...
    tn, J1, s1, PN, r1, c1, vr, Dt, FricFac);

end


function ContactPairs = AssembleSlipK(FEMod, ContactPairs, i, Na, N1a, N2a, Nb, N1b, N2b, ...
    Cur_n, Cur_N1Xa, Cur_N2Xa, Cur_N1Xb, Cur_N2Xb, tn, J1, s1, PN, r1, c1, vr, Dt, FricFac)

% --- Projection matrices ---
A_ab = [ Cur_N1Xa'*Cur_N1Xb, Cur_N1Xa'*Cur_N2Xb;
         Cur_N2Xa'*Cur_N1Xb, Cur_N2Xa'*Cur_N2Xb ];
a_ab = inv(A_ab);

g1_bar_slave  = a_ab(1,1)*Cur_N1Xb + a_ab(2,1)*Cur_N2Xb;
g2_bar_slave  = a_ab(1,2)*Cur_N1Xb + a_ab(2,2)*Cur_N2Xb;
g1_bar_master = a_ab(1,1)*Cur_N1Xa + a_ab(1,2)*Cur_N2Xa;
g2_bar_master = a_ab(2,1)*Cur_N1Xa + a_ab(2,2)*Cur_N2Xa;

N1  = Cur_n * Cur_n';
N1_bar = eye(3) - Cur_N1Xa*g1_bar_slave' - Cur_N2Xa*g2_bar_slave';

mc1_bar = kron(N1a', g1_bar_slave) + kron(N2a', g2_bar_slave);
mb2_bar = kron(N1b', g1_bar_master) + kron(N2b', g2_bar_master);

Cur_g1_hat_slave = TransVect2SkewSym(Cur_N1Xa);
Cur_g2_hat_slave = TransVect2SkewSym(Cur_N2Xa);
Ac = (kron(N1a', Cur_g2_hat_slave) - kron(N2a', Cur_g1_hat_slave)) / J1;

N1_wave = Cur_n*(N1_bar*Cur_n)';

% --- Mc1_bar & Mb2_bar as in original code ---
Mc1_bar = [Cur_n*mc1_bar(:,1)', Cur_n*mc1_bar(:,2)', Cur_n*mc1_bar(:,3)', Cur_n*mc1_bar(:,4)'];
Mb2_bar = [-Cur_n*mb2_bar(:,1)', -Cur_n*mb2_bar(:,2)', -Cur_n*mb2_bar(:,3)', -Cur_n*mb2_bar(:,4)'];

N12a = [N1a'; N2a'];
N12b = [N1b'; N2b'];
Gbc = ContactPairs.Cur_g(i) * (N12b' * a_ab * N12a);

% --- Frictionless stiffness ---
[FrictionlessK] = AssembleFrictionlessK(Na, Nb, ContactPairs.pc(i), tn, Ac, Mc1_bar, Mb2_bar, Gbc, N1, N1_wave, J1);

% --- Frictional stiffness ---
Frictional_K = zeros(24);
if FricFac ~= 0 && norm(s1) > 1e-8
    Frictional_K = AssembleFrictionalK(Na, Nb, ContactPairs.pc(i), tn, Ac, Cur_n, mc1_bar, mb2_bar, ...
        s1, PN, r1, c1, Dt, Cur_g1_hat_slave, Cur_g2_hat_slave, N1, N1_bar, J1, ContactPairs.Cur_g(i), Gbc);
end

% --- Total contact stiffness ---
ContactK = FrictionlessK + Frictional_K;

% --- Assemble into global stiffness ---
ContactDOF = [CurSlaveDOF(FEMod, ContactPairs, i); CurMasterDOF(FEMod, ContactPairs, i)];
GKF(ContactDOF, ContactDOF) = GKF(ContactDOF, ContactDOF) - ContactK;

end


function FrictionlessK = AssembleFrictionlessK(Na, Nb, pc, tn, Ac, Mc1_bar, Mb2_bar, Gbc, N1, N1_wave, J1)
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


function Frictional_K = AssembleFrictionalK(Na, Nb, pc, tn, Ac, Cur_n, mc1_bar, mb2_bar, ...
    s1, PN, r1, c1, Dt, Cur_g1_hat_slave, Cur_g2_hat_slave, N1, N1_bar, J1, Cur_g)

Q1 = ((Cur_n'*r1)*eye(3) + Cur_n*r1') / J1;
dg1_hat_slave = Cur_g1_hat_slave; % Already computed outside
dg2_hat_slave = Cur_g2_hat_slave;
Ac1_bar = (kron(Na', dg2_hat_slave) - kron(Na', dg1_hat_slave)) / J1;

dh = norm(PN* (r1/Dt)) * Dt;
Ps = (eye(3) - s1*s1') / dh;
R1 = ((Cur_n'*r1)*eye(3) + Cur_n*r1') / Cur_g;
B1 = (Ps*c1)*(N1_bar*Cur_n)' - Ps*PN;
L1 = Cur_g*Ps*(PN*Q1 + R1 - eye(3))*PN;

Jc1 = L1*Ac - Cur_g*Ps*PN*Ac1_bar;
hc1_add = N1*mc1_bar + Ac*kron(eye(4), Cur_n);
hc1_sub = N1*mc1_bar - Ac*kron(eye(4), Cur_n);
S1 = s1*Cur_n';
S1_wave = s1*(N1_bar*Cur_n)';

Frictional_K11 = zeros(12); Frictional_K12 = zeros(12);
Frictional_K21 = zeros(12); Frictional_K22 = zeros(12);

for aa=1:4
    for bb=1:4
        idxA = (3*aa-2):(3*aa); idxB=(3*bb-2):(3*bb);
        tempK = (-Na(aa)*Na(bb)*pc*(S1_wave) - Na(aa)*tn*(s1*hc1_sub(:,bb)' + Cur_g*Ps*c1*hc1_add(:,bb)'-Jc1(:,idxB)))*J1;
        Frictional_K11(idxA,idxB) = Frictional_K11(idxA,idxB) + tempK;
        tempK = (Na(aa)*Nb(bb)*pc*(S1_wave))*J1;
        Frictional_K12(idxA,idxB) = Frictional_K12(idxA,idxB) + tempK;
        tempK = (Nb(aa)*Na(bb)*pc*(S1_wave) + Nb(aa)*tn*(s1*hc1_sub(:,bb)' + Cur_g*Ps*c1*hc1_add(:,bb)'-Jc1(:,idxB)) ...
            + Na(bb)*tn*(-s1*mb2_bar(:,aa)') + Cur_g*tn*S1*aa*bb )*J1; % Gbc term
        Frictional_K21(idxA,idxB) = Frictional_K21(idxA,idxB) + tempK;
        tempK = (-Nb(aa)*Nb(bb)*pc*(S1_wave) - Nb(bb)*tn*(-s1*mb2_bar(:,aa)'))*J1;
        Frictional_K22(idxA,idxB) = Frictional_K22(idxA,idxB) + tempK;
    end
end

Frictional_K = [Frictional_K11, Frictional_K12; Frictional_K21, Frictional_K22];

end

function Stick_K = AssembleStickStiffness(Na, Nb, pc, tv, Ac, Cur_n, J1)
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

function [Cur_n, J1, Cur_N1Xa, Cur_N2Xa, Cur_x1] = ComputeSlaveSurfaceGeometry(Na, N1a, N2a, CurSlaveSurfXYZ)
Cur_x1   = sum(Na .* CurSlaveSurfXYZ,1)';
Cur_N1Xa = sum(N1a .* CurSlaveSurfXYZ,1)';
Cur_N2Xa = sum(N2a .* CurSlaveSurfXYZ,1)';
Cur_n = cross(Cur_N1Xa, Cur_N2Xa);
Cur_n = Cur_n / norm(Cur_n);
J1 = norm(cross(Cur_N1Xa, Cur_N2Xa));
end

function ContactNodeForce = AssembleContactForce(Na, Nb, J1, tv)
ContactNodeForce = zeros(24,1);
for a = 1:4
    ContactNodeForce((3*a-2):3*a)       = Na(a) * J1 * tv;
    ContactNodeForce(((3*a-2):3*a)+12) = -Nb(a) * J1 * tv;
end
end