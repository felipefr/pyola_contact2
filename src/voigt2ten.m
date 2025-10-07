function A=voigt2ten(v,op)
if op == 'strain'
    A = [v(1) 0.5*v(4) 0.5*v(6);0.5*v(4) v(2) 0.5*v(5);0.5*v(6) 0.5*v(5) v(3)];
else
    A = [v(1) v(4) v(6);v(4) v(2) v(5);v(6) v(5) v(3)];
end 
end 