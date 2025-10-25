%  ===  Shape function for 4-node surface  ===
function [N,N1,N2]=GetSurfaceShapeFunction(r,s)
N=[0.25*(r-1)*(s-1);-0.25*(r+1)*(s-1);0.25*(r+1)*(s+1);-0.25*(r-1)*(s+1)];
N1=[0.25*(s-1);-0.25*(s-1);0.25*(s+1);-0.25*(s+1)];N2=[0.25*(r-1);-0.25*(r+1);0.25*(r+1);-0.25*(r-1)];
end