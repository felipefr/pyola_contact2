function contactPairs = updateContact(contactPairs)
% === Update contact history between time steps (struct-of-vectors version) ===

nPairs = size(contactPairs.SlaveSurf, 2);

for i = 1:nPairs  % Loop over contact pairs
    if contactPairs.CurContactState(i) == 0
        % --- No contact ---
        contactPairs.PreMasterSurf(:, i) = [0; 0];
        contactPairs.rp(i) = 0;
        contactPairs.sp(i) = 0;
        contactPairs.PreContactState(i) = 0;
        contactPairs.Pre_g(i) = 0;
        contactPairs.Pressure(i) = 0;
        contactPairs.Traction(i) = 0;
    else
        % --- Slip or stick contact ---
        contactPairs.PreMasterSurf(:, i) = contactPairs.CurMasterSurf(:, i);
        contactPairs.rp(i) = contactPairs.rc(i);
        contactPairs.sp(i) = contactPairs.sc(i);
        contactPairs.PreContactState(i) = contactPairs.CurContactState(i);
        contactPairs.Pre_g(i) = contactPairs.Cur_g(i);
    end

    % --- Reset current step quantities ---
    contactPairs.rc(i) = 0;
    contactPairs.sc(i) = 0;
    contactPairs.Cur_g(i) = 0;
    contactPairs.CurMasterSurf(:, i) = [0; 0];
    contactPairs.CurContactState(i) = 0;
end
end
