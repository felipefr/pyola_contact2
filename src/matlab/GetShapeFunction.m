% =================== FEM UTILITIES ====================================
%  ===  Shape function for 8-node element  ===
function [ShpD, Det] = GetShapeFunction(XI, Elxy)
XNode=[-1  1  1 -1 -1  1  1 -1;-1 -1  1  1 -1 -1  1  1;-1 -1 -1 -1  1  1  1  1];
DSF=zeros(3,8);
for I=1:8
    XP = XNode(1,I);YP = XNode(2,I);ZP = XNode(3,I);XI0 = [1+XI(1)*XP 1+XI(2)*YP 1+XI(3)*ZP];
    DSF(1,I) = 0.125*XP*XI0(2)*XI0(3);DSF(2,I) = 0.125*YP*XI0(1)*XI0(3);DSF(3,I) = 0.125*ZP*XI0(1)*XI0(2);
end
GJ = DSF*Elxy;Det = det(GJ);GJinv=inv(GJ);ShpD=GJinv*DSF;
end