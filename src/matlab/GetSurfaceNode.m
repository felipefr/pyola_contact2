%  ===  Obtain surface node number  ===
function [SurfNode]=GetSurfaceNode(elementLE,SurfSign)
SurfNode=[];
if(SurfSign==1),SurfNode=elementLE(1,[4,3,2,1]);elseif(SurfSign==2),SurfNode=elementLE(1,[6,7,8,5]);
elseif(SurfSign==3),SurfNode=elementLE(1,[2,6,5,1]);elseif(SurfSign==4),SurfNode=elementLE(1,[2,3,7,6]);
elseif(SurfSign==5),SurfNode=elementLE(1,[3,4,8,7]);elseif(SurfSign==6),SurfNode=elementLE(1,[5,8,4,1]);
end
end