%  ===  Plot surface contours ===
function PlotContourSurface(FEMod,Component)
Difference=max(max(Component))-min(min(Component));
AVG=1;SurfaceCount=size(FEMod.SlaveSurf,2);
SlaveSurfNodes=zeros(SurfaceCount,4);
for i=1:size(FEMod.SlaveSurf,2)
    SlaveSurfNodes(i,:)=GetSurfaceNode(FEMod.Eles(FEMod.SlaveSurf(1,i),:),FEMod.SlaveSurf(2,i));
end
allSalveNodes=unique(SlaveSurfNodes(:));
for i=1:size(allSalveNodes,1)
    itemp=(SlaveSurfNodes==allSalveNodes(i));
    Cut=max(Component(itemp))-min(Component(itemp));
    if 0<Cut&&Cut<=AVG*Difference
        Component(itemp)=mean(Component(itemp));
    end
end
myColor=1/255*[0,0,255;  0,70,255;   0,141,255;  0,212,255;
    0,255,141;  141,255,0;  255,212,0;  255,141,0;
    255,70,0;  255,0,0;];
fm=[1,2,3,4];xyz=cell(1,SurfaceCount);profile=cell(1,SurfaceCount);
for i=1:SurfaceCount
    xyz{i}=FEMod.Nodes(SlaveSurfNodes(i,:),:);profile{i}=Component(i,:)';
end
figure;
aaa=cellfun(@patch,repmat({'Vertices'},1,SurfaceCount),xyz,...
    repmat({'Faces'},1,SurfaceCount),repmat({fm},1,SurfaceCount),...
    repmat({'FaceVertexCdata'},1,SurfaceCount),profile,...
    repmat({'FaceColor'},1,SurfaceCount),repmat({'interp'},1,SurfaceCount));
view([0 0]);rotate3d on;axis off;
colormap(myColor);caxis([min(min(Component)),max(max(Component))]);
t1=caxis;t1=linspace(t1(1),t1(2),13);
colorbar('ytick',t1,'Location','westoutside');axis equal;
temp=FEMod.Nodes(SlaveSurfNodes(:),:);
xlim([min(temp(:,1))-1 max(temp(:,1))+1])
ylim([min(temp(:,2))-1 max(temp(:,2))+1])
zlim([min(temp(:,3))-1 max(temp(:,3))+1])
end
