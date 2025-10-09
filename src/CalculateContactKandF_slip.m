function [GKF, Residual, ContactPairs] = CalculateContactKandF_slip(FEMod, ContactPairs, Dt, ...
    PreDisp, i, GKF, Residual, Disp, IntegralPoint)

FricFac = FEMod.FricFac;

tn = ContactPairs.Cur_g(i) * ContactPairs.pc(i);

% --- current slave geometry & previous slave geometry ---
CurIP = IntegralPoint(ContactPairs.SlaveIntegralPoint(i), :)';
[Na, N1a, N2a] = GetSurfaceShapeFunction(CurIP(1), CurIP(2));
[CurSlaveSurfXYZ, SlaveSurfDOF]     = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs.SlaveSurf(:, i));
[PreSlaveSurfNodeXYZ, ~]            = GetSurfaceNodeLocation(FEMod, PreDisp, ContactPairs.SlaveSurf(:, i));
 
 
Cur_x1 = sum(Na .* CurSlaveSurfXYZ, 1)';  Pre_x1 = sum(Na .* PreSlaveSurfNodeXYZ, 1)';
dx1 = Cur_x1 - Pre_x1;

Pre_N1Xa = sum(N1a .* PreSlaveSurfNodeXYZ, 1)'; Pre_N2Xa = sum(N2a .* PreSlaveSurfNodeXYZ, 1)';
Cur_N1Xa = sum(N1a .* CurSlaveSurfXYZ, 1)';   Cur_N2Xa = sum(N2a .* CurSlaveSurfXYZ, 1)';

Cur_n = cross(Cur_N1Xa, Cur_N2Xa); 
Cur_n = Cur_n / norm(Cur_n);

J1 = norm(cross(Cur_N1Xa, Cur_N2Xa));
PN = eye(3) - Cur_n * Cur_n';

dg1_slave = Cur_N1Xa - Pre_N1Xa;
dg2_slave = Cur_N2Xa - Pre_N2Xa;
m1 = cross(dg1_slave, Cur_N2Xa) + cross(Cur_N1Xa, dg2_slave);
c1 = PN * m1 / J1;

% --- master geometry at current and previous steps ---
[Nb, N1b, N2b] = GetSurfaceShapeFunction(ContactPairs.rc(i), ContactPairs.sc(i));
[CurMasterSurfNodeXYZ, MasterSurfDOF] = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs.CurMasterSurf(:, i));
[PreMasterSurfNodeXYZ, ~]             = GetSurfaceNodeLocation(FEMod, PreDisp, ContactPairs.CurMasterSurf(:, i));

Cur_x2 = sum(Nb .* CurMasterSurfNodeXYZ, 1)';   Pre_x2 = sum(Nb .* PreMasterSurfNodeXYZ, 1)';
Cur_N1Xb = sum(N1b .* CurMasterSurfNodeXYZ, 1)'; Cur_N2Xb = sum(N2b .* CurMasterSurfNodeXYZ, 1)';
dx2 = Cur_x2 - Pre_x2;
% 
% % --- relative velocity and tangential direction ---
r1 = ContactPairs.Cur_g(i) * c1 + dx1 - dx2;
vr = r1 / Dt;
s1_temp = PN * vr;

if sqrt(s1_temp' * s1_temp) > 1e-8
    s1 = s1_temp / sqrt(s1_temp' * s1_temp);
else
    s1 = zeros(3, 1);
    dh = 0;  
end

% --- contact nodal force (with friction) ---
ContactNodeForce = zeros(24, 1);
tv = tn * (Cur_n + FricFac * s1);
temp_f1a = tv * J1;

ContactPairs.Pressure(i)  = abs(tn);
ContactPairs.Traction(i) = abs(sqrt(tv' * tv));  % save pressure/traction

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
Gbc = ContactPairs.Cur_g(i) * (N12b' * a_ab * N12a);

% % --- frictionless stiffness baseline ---

FrictionlessK = getFrictionlessK(i, Gbc, J1, Ac, N1, Na, Nb, N1_wave, tn, Mc1_bar, Mb2_bar, ContactPairs);
GKF(ContactDOF, ContactDOF) = GKF(ContactDOF, ContactDOF) - FrictionlessK;

% % --- frictional additions (only if friction != 0 and s1 nonzero) ---
% if FricFac ~= 0 && sqrt(s1' * s1) > 1e-8
%     Frictional_K = getFrictionalK(i, Gbc, Ac, tn, r1, c1, s1, m1, Cur_n, J1, dg1_slave, ... 
%         dg2_slave, PN, vr, Dt, mc1_bar, mb2_bar, N1, Na, Nb, N1a, N2a, N1_bar, ContactPairs, FricFac);
%     GKF(ContactDOF, ContactDOF) = GKF(ContactDOF, ContactDOF) - Frictional_K;
% end

% structAux = struct( ...
%     'Gbc', Gbc, ...
%     'J1', J1, ...
%     'Ac', Ac, ...
%     'N1', N1, ...
%     'Na', Na, ...
%     'Nb', Nb, ...
%     'N1_wave', N1_wave, ...
%     'tn', tn, ...
%     'Mc1_bar', Mc1_bar, ...
%     'Mb2_bar', Mb2_bar, ...
%     'r1', r1, ...
%     'c1', c1, ...
%     's1', s1, ...
%     'm1', m1, ...
%     'Cur_n', Cur_n, ...
%     'dg1_slave', dg1_slave, ...
%     'dg2_slave', dg2_slave, ...
%     'PN', PN, ...
%     'vr', vr, ...
%     'Dt', Dt, ...
%     'mc1_bar', mc1_bar, ...
%     'mb2_bar', mb2_bar, ...
%     'N1a', N1a, ...
%     'N2a', N2a, ...
%     'N1_bar', N1_bar ...
% );


end