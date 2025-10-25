%  ===  Obtain surface node location and surface node DOFs  ===
function [SurfNodeXYZ,SurfNodeDOF]=GetSurfaceNodeLocation(FEMod,Disp,Surf)
[SurfNode]=GetSurfaceNode(FEMod.Eles(Surf(1),:),Surf(2));SurfNodeDOF=zeros(size(SurfNode,2)*3,1);
for m=1:size(SurfNode,2), SurfNodeDOF((3*m-2):3*m,1)=[(3*SurfNode(m)-2):3*SurfNode(m)]'; end
SurfNodeDis=reshape(Disp(SurfNodeDOF),3,size(SurfNode,2))';
SurfNodeXYZ=FEMod.Nodes(SurfNode,:)+SurfNodeDis;
end