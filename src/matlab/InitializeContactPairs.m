function ContactPairs = InitializeContactPairs(FEMod)
% InitializeContactPairs - Initialize contact data as a struct of arrays
%
% Input:
%   FEMod - structure with FEM data, must include SlaveSurf (2×n)
%
% Output:
%   ContactPairs - structure where each field is a vector or matrix
%                  (not a vector of structs)

nSlave = size(FEMod.SlaveSurf, 2);
nGauss = 4;
nPairs = nSlave * nGauss;

% Preallocate arrays
ContactPairs.pc  = 1e6 * ones(1, nPairs);      % scalar constant per pair
ContactPairs.SlaveSurf = zeros(2, nPairs);     % 2×nPairs
ContactPairs.SlaveIntegralPoint = zeros(1, nPairs);

ContactPairs.CurMasterSurf = zeros(2, nPairs);
ContactPairs.rc = zeros(1, nPairs);
ContactPairs.sc = zeros(1, nPairs);
ContactPairs.Cur_g = zeros(1, nPairs);
ContactPairs.Pre_g = zeros(1, nPairs);
ContactPairs.PreMasterSurf = zeros(2, nPairs);
ContactPairs.rp = zeros(1, nPairs);
ContactPairs.sp = zeros(1, nPairs);
ContactPairs.CurContactState = zeros(1, nPairs);
ContactPairs.PreContactState = zeros(1, nPairs);
ContactPairs.Pressure = zeros(1, nPairs);
ContactPairs.Traction = zeros(1, nPairs);

% Fill fields
for i = 1:nSlave
    for j = 1:nGauss
        k = (i-1)*nGauss + j;
        ContactPairs.SlaveSurf(:,k) = FEMod.SlaveSurf(:,i);
        ContactPairs.SlaveIntegralPoint(k) = j;
    end
end
end
