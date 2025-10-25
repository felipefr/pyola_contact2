function v=ten2voigt(A,op)
if op == 'strain'
    v = [A(1,1) A(2,2) A(3,3) 2*A(1,2) 2*A(2,3) 2*A(1,3)]';
else
    v = [A(1,1) A(2,2) A(3,3) A(1,2) A(2,3) A(1,3)]';
end 
end 