function [Frictional_K] = getFrictionalK(i, ContactPairs, FricFac, structAux)

Gbc       = structAux.Gbc;
Ac        = structAux.Ac;
tn        = structAux.tn;
r1        = structAux.r1;
c1        = structAux.c1;
s1        = structAux.s1;
m1        = structAux.m1;
Cur_n     = structAux.Cur_n;
J1        = structAux.J1;
dg1_slave = structAux.dg1_slave;
dg2_slave = structAux.dg2_slave;
PN        = structAux.PN;
vr        = structAux.vr;
Dt        = structAux.Dt;
mc1_bar   = structAux.mc1_bar;
mb2_bar   = structAux.mb2_bar;
N1        = structAux.N1;
Na        = structAux.Na;
Nb        = structAux.Nb;
N1a       = structAux.N1a;
N2a       = structAux.N2a;
N1_bar    = structAux.N1_bar;

Frictional_K = zeros(24);

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