%  ===  Plot displacement and stress contours  ===
function PlotStructuralContours(Nodes,Elements,U,Component)
NodeCount = size(Nodes,1) ;ElementCount = size(Elements,1) ;
ElementNodeCount=8;
value = zeros(ElementNodeCount,ElementCount) ;
if size(Component,1)>1
    for i=1:ElementCount,nd=Elements(i,:);value(:,i) = Component(nd) ;end
else
    Difference=max(Component)-min(Component);AVG=1;
    for i=1:1:NodeCount
        TElements=Elements';itemp=(TElements==i);
        Cut=max(Component(1,itemp))-min(Component(1,itemp));
        if 0<Cut&&Cut<=AVG*Difference(1)
            Component(1,itemp)=mean(Component(1,itemp));
        end
    end
    value=reshape(Component,ElementNodeCount,ElementCount);
end
myColor=1/255*[0,0,255;  0,70,255;   0,141,255;  0,212,255;
    0,255,141;  141,255,0;  255,212,0;  255,141,0;
    255,70,0;  255,0,0;];
newNodes=Nodes';newNodes=newNodes(:);newNodes=newNodes+U;
newNodes=reshape(newNodes,[3,size(Nodes,1)]);newNodes=newNodes';
fm = [1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8; 1 2 3 4; 5 6 7 8];
xyz = cell(1,ElementCount) ;profile = xyz ;
for e=1:ElementCount
    nd=Elements(e,:);X = newNodes(nd,1) ;Y = newNodes(nd,2) ;
    Z = newNodes(nd,3) ;xyz{e} = [X Y Z] ;profile{e} = value(:,e);
end
figure;
aaa=cellfun(@patch,repmat({'Vertices'},1,ElementCount),xyz,.......
    repmat({'Faces'},1,ElementCount),repmat({fm},1,ElementCount),...... 
    repmat({'FaceVertexCdata'},1,ElementCount),profile,......
    repmat({'FaceColor'},1,ElementCount),repmat({'interp'},1,ElementCount));
view([180 -80]);rotate3d on;axis off;
colormap(myColor);caxis([min(Component),max(Component)]);
t1=caxis;t1=linspace(t1(1),t1(2),13);
colorbar('ytick',t1,'Location','westoutside');
axis equal;rotate(aaa,[0,1,0],-5,[0,0,0]);
end