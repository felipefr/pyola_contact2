%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% These MATLAB codes are written by Bin Wang, Wenjie Zuo 
% Email: zuowenjie@jlu.edu.cn
% It is an educational MATLAB implementation of contact problem
% This MATLAB code can be downloaded from https://www.researchgate.net/profile/Wenjie-Zuo-3 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  ===  Main function  ===
function ContactFEA_refac()
% Mesh, solver params, etc
[FEMod]= ModelInformation_Beam();
Dt=0.01; MinDt=1.0E-7; IterMax=16; GivenIter=8; MaxDt=0.1; Time = 0;% N-R parameters

% Material
E=FEMod.Prop(1); nu=FEMod.Prop(2);
Dtan= getIsotropicCelas(E,nu);

contactPairs=InitializeContactPairs(FEMod);

[NodeNum, Dim] = size(FEMod.Nodes); AllDOF = Dim*NodeNum;Disp=zeros(AllDOF,1);
IterOld=GivenIter+1; NRConvergeNum=0; Istep = -1; Flag10 = 1;

% Main loop
while(Flag10 == 1)%Incremental loop
    Flag10 = 0; Flag11 = 1;Flag20 = 1;ReductionNumber = 0;
    DispSave = Disp; tempContactPairs=contactPairs;	Time0 = Time;
    Istep = Istep + 1; Time = Time + Dt;
    while(Flag11 == 1)%Reduction loop
        NRConvergeNum=NRConvergeNum+1;  Flag11 = 0;
        if ((Time-1)>1E-10)%Check whether the calculation is completed
            if ((1+Dt-Time)>1E-10),	Dt=1+Dt-Time; Time=1;else, break; end
        end
        Factor =Time;  SDisp = Dt*FEMod.Cons(:,3);  Iter = 0;PreDisp=Disp;
        while(Flag20 == 1)%NR loop
            Flag20 = 0; Iter = Iter + 1;
            Residual=zeros(AllDOF,1);GKF = sparse(AllDOF,AllDOF);
            ExtFVect=zeros(AllDOF,1); NCon=size(FEMod.Cons,1);
            [Residual,GKF]=GetStiffnessAndForce(FEMod,Disp,Residual,GKF,Dtan);%Internal force and tangent stiffness matrix
       
            [contactPairs,GKF,Residual]=DetermineContactState(FEMod,...
                contactPairs,Dt,PreDisp,GKF,Residual,Disp);
            if size(FEMod.ExtF,1)>0, LOC = Dim*(FEMod.ExtF(:,1)-1)+FEMod.ExtF(:,2);
                ExtFVect(LOC) = ExtFVect(LOC) + Factor*FEMod.ExtF(:,3); end%Load boundary
            Residual=Residual+ExtFVect;
            if NCon~=0%Displacement boundary
                FixDOF=Dim*(FEMod.Cons(:,1)-1)+FEMod.Cons(:,2); GKF(FixDOF,:)=zeros(NCon,AllDOF);
                GKF(FixDOF,FixDOF)=eye(NCon); Residual(FixDOF)=0;
                if Iter==1, Residual(FixDOF) = SDisp(:); end
            end
            if(Iter>1)
                FixDOF=Dim*(FEMod.Cons(:,1)-1)+FEMod.Cons(:,2); FreeDOF=setdiff(1:AllDOF,FixDOF);
                [Resid,~]=max(abs(Residual(FreeDOF)));
                if Iter>2, fprintf(1,'%27d %14.5e \n',Iter,full(Resid));
                else, fprintf(1,'\n \t Time  Time step   Iter \t  Residual \n');
                    fprintf(1,'%10.5f %10.3e %5d %14.5e \n',Time,Dt,Iter,full(Resid)); end
                if(Resid<1e-7)%Determine whether it converges
                    updateContact(contactPairs)
                    if NRConvergeNum > 1 && Iter < GivenIter &&IterOld < GivenIter
                        Enlarge = 1.5; Dt = min(Enlarge * Dt, MaxDt);%Increase the time increment
                    end
                    IterOld=Iter; Flag10 = 1; break;
                end
                if (Iter+1>IterMax)%Too many NR iteration
                    Reduce=0.25; Dt = Reduce*Dt; Time = Time0 + Dt;%decrease time increment
                    if (Dt<MinDt), error('Incremental step too small'); return; end
                    Disp=DispSave; contactPairs=tempContactPairs;
                    fprintf(1,'Not converged or reached the MaxIteration. Reducing load increment %3d\n',ReductionNumber);
                    NRConvergeNum=0; Flag11 = 1; Flag20 = 1; break;
                end
            end
            IncreDisp = GKF\Residual; Disp = Disp + IncreDisp; Flag20 = 1;
        end
    end
end

% Plotting

% Plot Displacement magnitude
UM=(Disp([1:3:size(Disp,1)]).^2+Disp([2:3:size(Disp,1)]).^2+Disp([3:3:size(Disp,1)]).^2).^0.5;
PlotStructuralContours(FEMod.Nodes,FEMod.Eles,Disp,UM);title("Umag");

% Plot Von Mises
[NodeCauchyMises,NodeCauchy]=CalculateCauchyStress(FEMod,Disp,Dtan);
PlotStructuralContours(FEMod.Nodes,FEMod.Eles,Disp,NodeCauchyMises);title("CauchyMises")

% Plot Contact
PlotContactPressureAndTraction(FEMod,contactPairs)

%Plot contours Displacement
for i=1:3,PlotStructuralContours(FEMod.Nodes,FEMod.Eles,Disp,Disp([i:3:size(Disp,1)]));title("U"+num2str(i));end

% Plot contours stress
for i=1:6,PlotStructuralContours(FEMod.Nodes,FEMod.Eles,Disp,NodeCauchy(i,:));title("CauchyStress"+num2str(i));end

end

% ===== POST PROCESSING =============================================

%  ===  Obtain node Cauchy stress  ===
function [NodeCauchyMises,NodeCauchy]=CalculateCauchyStress(FEMod,Disp,Dtan)
XG=[-0.57735026918963D0, 0.57735026918963D0];Index=0;
NodeCauchyMises=zeros(1,8*size(FEMod.Eles,1));NodeCauchy=zeros(6,8*size(FEMod.Eles,1));
for IE=1:size(FEMod.Eles,1)
    Elxy=FEMod.Nodes(FEMod.Eles(IE,:),:);IDOF=zeros(1,24);
    for I=1:8,II=(I-1)*3+1;IDOF(II:II+2)=(FEMod.Eles(IE,I)-1)*3+1:(FEMod.Eles(IE,I)-1)*3+3;end
    EleDisp=Disp(IDOF);EleDisp=reshape(EleDisp,3,8);tempStress=zeros(6,1);
    for LX=1:2, for LY=1:2, for LZ=1:2
                E1=XG(LX); E2=XG(LY); E3=XG(LZ);Index=Index+1;
                [Shpd, ~] = GetShapeFunction([E1 E2 E3], Elxy);F=EleDisp*Shpd' + eye(3);
                Strain=0.5*(F'*F-eye(3));   % Calculate Lagrangian strain
                Strain=[Strain(1,1) Strain(2,2) Strain(3,3) 2*Strain(1,2) 2*Strain(2,3) 2*Strain(1,3)]';
                Stress2PK=Dtan*Strain;temp= TransPK2toCauchy(F, Stress2PK);% Calculate Cauchy stress
                tempStress=tempStress+temp;
            end; end; end
    NodeCauchy(1:6,(8*IE-7):8*IE)=(repmat(tempStress/8,1,8));
end
for i=1:size(NodeCauchyMises,2)%Calculate Cauchy Mises stress
    NodeCauchyMises(i)=sqrt(0.5*((NodeCauchy(1,i)-NodeCauchy(2,i))^2+(NodeCauchy(2,i)-NodeCauchy(3,i))^2+...
        (NodeCauchy(3,i)-NodeCauchy(1,i))^2)+3*(NodeCauchy(4,i)^2+NodeCauchy(5,i)^2+NodeCauchy(6,i)^2));
end
end


% =========== PLOT FUNCTIONS ===========================================
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

% ======================== CONTACT FUNCTIONS ============================
%  ===  Obtain contact pressure and traction  ===
function PlotContactPressureAndTraction(FEMod,ContactPairs)
GvalPressure=zeros(size(ContactPairs,1)/4,4);GvalTraction=GvalPressure;%n*4
for i=1:size(ContactPairs,1)/4
    GvalPressure(i,:)=[ContactPairs(4*i-3).Pressure,ContactPairs(4*i-2).Pressure,...
        ContactPairs(4*i-1).Pressure,ContactPairs(4*i).Pressure];
    GvalTraction(i,:)=[ContactPairs(4*i-3).Traction,ContactPairs(4*i-2).Traction,...
        ContactPairs(4*i-1).Traction,ContactPairs(4*i).Traction];
end
temp=GvalPressure;temp(find(temp>0))=1;temp=sum(temp,2);
ElCenterVal=sum(GvalPressure,2)./temp;ElCenterVal(find(isnan(ElCenterVal)==1))=0;%surface center value
NodeValPressure=repmat(ElCenterVal,1,4);
temp=GvalTraction;temp(find(temp>0))=1;temp=sum(temp,2);
ElCenterVal=sum(GvalTraction,2)./temp;ElCenterVal(find(isnan(ElCenterVal)==1))=0;%surface center value
NodeValTraction=repmat(ElCenterVal,1,4);
PlotContourSurface(FEMod,NodeValPressure);title("ContactPressure")
PlotContourSurface(FEMod,NodeValTraction);title("ContactTraction")
end






