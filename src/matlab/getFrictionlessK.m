function [FrictionlessK] = getFrictionlessK(i, Gbc, J1, Ac, N1, Na, Nb, N1_wave, tn, Mc1_bar, Mb2_bar, ContactPairs)
Frictionless_K11 = zeros(12); Frictionless_K12 = zeros(12);
Frictionless_K21 = zeros(12); Frictionless_K22 = zeros(12);

for aa = 1:4
    for bb = 1:4
        idxA = (3*aa-2):(3*aa); idxB = (3*bb-2):(3*bb);

        tempK = ( -Na(aa) * Na(bb) * ContactPairs.pc(i) * N1_wave ...
                  - Na(aa) * tn * (Ac(:, idxB) + Mc1_bar(:, idxB) * N1) ) * J1;
        Frictionless_K11(idxA, idxB) = Frictionless_K11(idxA, idxB) + tempK;

        tempK = ( Na(aa) * Nb(bb) * ContactPairs.pc(i) * N1_wave ) * J1;
        Frictionless_K12(idxA, idxB) = Frictionless_K12(idxA, idxB) + tempK;

        tempK = ( Nb(aa) * Na(bb) * ContactPairs.pc(i) * N1_wave ...
                  + Nb(aa) * tn * (Ac(:, idxB) + Mc1_bar(:, idxB) * N1) ...
                  + Na(bb) * tn * Mb2_bar(:, idxA) + Gbc(aa, bb) * tn * N1 ) * J1;
        Frictionless_K21(idxA, idxB) = Frictionless_K21(idxA, idxB) + tempK;

        tempK = ( -Nb(aa) * Nb(bb) * ContactPairs.pc(i) * N1_wave ...
                  - Nb(bb) * tn * Mb2_bar(:, idxA) ) * J1;
        Frictionless_K22(idxA, idxB) = Frictionless_K22(idxA, idxB) + tempK;
    end
end

FrictionlessK = [Frictionless_K11, Frictionless_K12; Frictionless_K21, Frictionless_K22];

% Gbc      = structAux.Gbc;
% J1       = structAux.J1;
% Ac       = structAux.Ac;
% N1       = structAux.N1;
% Na       = structAux.Na;
% Nb       = structAux.Nb;
% N1_wave  = structAux.N1_wave;
% tn       = structAux.tn;
% Mc1_bar  = structAux.Mc1_bar;
% Mb2_bar  = structAux.Mb2_bar;


end