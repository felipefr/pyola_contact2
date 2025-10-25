%  ===  Obtain skew-symmetric tensor of vector  ===
function [SkewSym]=TransVect2SkewSym(Vect)
SkewSym=[0,-Vect(3),Vect(2);Vect(3),0,-Vect(1);-Vect(2),Vect(1),0];
end
