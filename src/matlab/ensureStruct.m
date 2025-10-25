function s = ensureStruct(x)
%ENSURESTRUCT Convert cell array of structs to struct array if needed
%
%   s = ENSURESTRUCT(x) checks if x is a cell array. 
%   If so, it converts it to a struct array using cell2mat.
%   Otherwise, it returns x unchanged.

    if iscell(x)
        try
            s = cell2mat(x);
        catch ME
            warning('Failed to convert cell array to struct array: %s', ME.message);
            s = x; % fallback: keep as-is
        end
    else
        s = x;
    end
end