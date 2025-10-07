%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% These MATLAB codes are written by Bin Wang, Wenjie Zuo 
% Email: zuowenjie@jlu.edu.cn
% It is an educational MATLAB implementation of contact problem
% This MATLAB code can be downloaded from https://www.researchgate.net/profile/Wenjie-Zuo-3 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  ===  Main function  ===
function ContactFEA()
[FEMod]= ModelInformation_Beam();
ContactContactPairs=InitializeContactPairs(FEMod);
E=FEMod.Prop(1);u=FEMod.Prop(2);
Dtan=(E*(1-u))/((1+u)*(1-2*u))*[1 u/(1-u) u/(1-u) 0 0 0;u/(1-u) 1 u/(1-u) 0 0 0;
    u/(1-u) u/(1-u) 1 0 0 0;0 0 0 (1-2*u)/(2*(1-u)) 0 0;
    0 0 0 0 (1-2*u)/(2*(1-u)) 0;0 0 0 0 0 (1-2*u)/(2*(1-u))];
Dt=0.01; MinDt=1.0E-7; IterMax=16; GivenIter=8; MaxDt=0.1; Time = 0;% N-R parameters
[NodeNum, Dim] = size(FEMod.Nodes); AllDOF = Dim*NodeNum;Disp=zeros(AllDOF,1);
IterOld=GivenIter+1; NRConvergeNum=0; Istep = -1; Flag10 = 1;
while(Flag10 == 1)%Incremental loop
    Flag10 = 0; Flag11 = 1;Flag20 = 1;ReductionNumber = 0;
    DispSave = Disp; tempContactPairs=ContactContactPairs;	Time0 = Time;
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
            [ContactContactPairs,GKF,Residual]=DetermineContactState(FEMod,...
                ContactContactPairs,Dt,PreDisp,GKF,Residual,Disp);
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
                    for im=1:size(ContactContactPairs,1)%Update contact information
                        if ContactContactPairs(im).CurContactState==0%no contact
                            ContactContactPairs(im).PreMasterSurf=[0;0]; ContactContactPairs(im).rp=0; ContactContactPairs(im).sp=0;
                            ContactContactPairs(im).PreContactState=0; ContactContactPairs(im).Pre_g=0;
                            ContactContactPairs(im).Pressure=0;ContactContactPairs(im).Traction=0;
                        else%slip or stick contact
                            ContactContactPairs(im).PreMasterSurf=ContactContactPairs(im).CurMasterSurf;
                            ContactContactPairs(im).rp=ContactContactPairs(im).rc; ContactContactPairs(im).sp=ContactContactPairs(im).sc;
                            ContactContactPairs(im).PreContactState=ContactContactPairs(im).CurContactState;
                            ContactContactPairs(im).Pre_g=ContactContactPairs(im).Cur_g;
                        end
                        ContactContactPairs(im).rc=0; ContactContactPairs(im).sc=0;
                        ContactContactPairs(im).Cur_g=0;ContactContactPairs(im).CurMasterSurf=[0;0];
                        ContactContactPairs(im).CurContactState=0;
                    end
                    if NRConvergeNum > 1 && Iter < GivenIter &&IterOld < GivenIter
                        Enlarge = 1.5; Dt = min(Enlarge * Dt, MaxDt);%Increase the time increment
                    end
                    IterOld=Iter; Flag10 = 1; break;
                end
                if (Iter+1>IterMax)%Too many NR iteration
                    Reduce=0.25; Dt = Reduce*Dt; Time = Time0 + Dt;%decrease time increment
                    if (Dt<MinDt), error('Incremental step too small'); return; end
                    Disp=DispSave; ContactContactPairs=tempContactPairs;
                    fprintf(1,'Not converged or reached the MaxIteration. Reducing load increment %3d\n',ReductionNumber);
                    NRConvergeNum=0; Flag11 = 1; Flag20 = 1; break;
                end
            end
            IncreDisp = GKF\Residual; Disp = Disp + IncreDisp; Flag20 = 1;
        end
    end
end
for i=1:3,PlotStructuralContours(FEMod.Nodes,FEMod.Eles,Disp,Disp([i:3:size(Disp,1)]));title("U"+num2str(i));end%Plot contours
UM=(Disp([1:3:size(Disp,1)]).^2+Disp([2:3:size(Disp,1)]).^2+Disp([3:3:size(Disp,1)]).^2).^0.5;
PlotStructuralContours(FEMod.Nodes,FEMod.Eles,Disp,UM);title("Umag");
[NodeCauchyMises,NodeCauchy]=CalculateCauchyStress(FEMod,Disp,Dtan);
PlotStructuralContours(FEMod.Nodes,FEMod.Eles,Disp,NodeCauchyMises);title("CauchyMises")
for i=1:6,PlotStructuralContours(FEMod.Nodes,FEMod.Eles,Disp,NodeCauchy(i,:));title("CauchyStress"+num2str(i));end
PlotContactPressureAndTraction(FEMod,ContactContactPairs)
end
%  ===  Obtain global stiffness matrix and internal force vector ===
function [Resdual,GKF]=GetStiffnessAndForce(FEMod,Disp,Resdual,GKF,Dtan)
XG=[-0.57735026918963D0, 0.57735026918963D0];WGT=[1.00000000000000D0, 1.00000000000000D0];
for IE=1:size(FEMod.Eles,1)
    Elxy=FEMod.Nodes(FEMod.Eles(IE,:),:);IDOF=zeros(1,24);
    for I=1:8,II=(I-1)*3+1;IDOF(II:II+2)=(FEMod.Eles(IE,I)-1)*3+1:(FEMod.Eles(IE,I)-1)*3+3;end
    EleDisp=Disp(IDOF);EleDisp=reshape(EleDisp,3,8);%Element dispacement
    for LX=1:2, for LY=1:2, for LZ=1:2
                E1=XG(LX); E2=XG(LY); E3=XG(LZ);
                [Shpd, Det] = GetShapeFunction([E1 E2 E3], Elxy);
                FAC=WGT(LX)*WGT(LY)*WGT(LZ)*Det;
                F=EleDisp*Shpd' + eye(3);
                Strain=0.5*(F'*F-eye(3));   % Calculate Lagrangian strain
                Strain=[Strain(1,1) Strain(2,2) Strain(3,3) 2*Strain(1,2) 2*Strain(2,3) 2*Strain(1,3)]';
                Stress=Dtan*Strain;%PK2stress
                BN=zeros(6,24);BG=zeros(9,24);
                for I=1:8
                    COL=(I-1)*3+1:(I-1)*3+3;
                    BN(:,COL)=[Shpd(1,I)*F(1,1) Shpd(1,I)*F(2,1) Shpd(1,I)*F(3,1);
                        Shpd(2,I)*F(1,2) Shpd(2,I)*F(2,2) Shpd(2,I)*F(3,2);
                        Shpd(3,I)*F(1,3) Shpd(3,I)*F(2,3) Shpd(3,I)*F(3,3);
                        Shpd(1,I)*F(1,2)+Shpd(2,I)*F(1,1) Shpd(1,I)*F(2,2)+Shpd(2,I)*F(2,1) Shpd(1,I)*F(3,2)+Shpd(2,I)*F(3,1);
                        Shpd(2,I)*F(1,3)+Shpd(3,I)*F(1,2) Shpd(2,I)*F(2,3)+Shpd(3,I)*F(2,2) Shpd(2,I)*F(3,3)+Shpd(3,I)*F(3,2);
                        Shpd(1,I)*F(1,3)+Shpd(3,I)*F(1,1) Shpd(1,I)*F(2,3)+Shpd(3,I)*F(2,1) Shpd(1,I)*F(3,3)+Shpd(3,I)*F(3,1)];
                    BG(:,COL)=[Shpd(1,I) 0         0;Shpd(2,I) 0         0;Shpd(3,I) 0         0;
                        0         Shpd(1,I) 0;0         Shpd(2,I) 0;0         Shpd(3,I) 0;
                        0         0         Shpd(1,I);0         0         Shpd(2,I);0         0         Shpd(3,I)];
                end
                Resdual(IDOF) = Resdual(IDOF) - FAC*BN'*Stress;%Assemble internal force vector
                SIG=[Stress(1) Stress(4) Stress(6);Stress(4) Stress(2) Stress(5);Stress(6) Stress(5) Stress(3)];
                SHEAD=zeros(9);SHEAD(1:3,1:3)=SIG;SHEAD(4:6,4:6)=SIG;SHEAD(7:9,7:9)=SIG;
                EKF = BN'*Dtan*BN + BG'*SHEAD*BG;
                GKF(IDOF,IDOF)=GKF(IDOF,IDOF)+FAC*EKF;%Assemble tangent stiffness matrix
            end; end; end
end
end
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
%  ===  Transform PK2 stress into Cauchy stress  ===
function Cauchy=TransPK2toCauchy(F, StressPK2)
PK=[StressPK2(1) StressPK2(4) StressPK2(6);StressPK2(4) StressPK2(2) StressPK2(5);StressPK2(6) StressPK2(5) StressPK2(3)];
DETF = det(F);ST = F*PK*F'/DETF; Cauchy=[ST(1,1) ST(2,2) ST(3,3) ST(1,2) ST(2,3) ST(1,3)]';
end
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
%  ===  Initialize contact pair information  ===
function ContactPairs=InitializeContactPairs(FEMod)
ContactPair.pc=1e6; ContactPair.SlaveSurf=[0;0];ContactPair.SlaveIntegralPoint=0;
ContactPair.CurMasterSurf=[0;0];ContactPair.rc=0;ContactPair.sc=0;
ContactPair.Cur_g=0;ContactPair.Pre_g=0;ContactPair.PreMasterSurf=[0;0];
ContactPair.rp=0;ContactPair.sp=0;ContactPair.CurContactState=0;ContactPair.PreContactState=0;
ContactPair.Pressure=0;ContactPair.Traction=0;
ContactPairs = repmat(ContactPair, size(FEMod.SlaveSurf, 2) * 4, 1);
for i=1:size(FEMod.SlaveSurf,2)
    ContactPairs(4*i-3).SlaveSurf=FEMod.SlaveSurf(:,i); ContactPairs(4*i-3).SlaveIntegralPoint=1;
    ContactPairs(4*i-2).SlaveSurf=FEMod.SlaveSurf(:,i); ContactPairs(4*i-2).SlaveIntegralPoint=2;
    ContactPairs(4*i-1).SlaveSurf=FEMod.SlaveSurf(:,i); ContactPairs(4*i-1).SlaveIntegralPoint=3;
    ContactPairs(4*i).SlaveSurf=FEMod.SlaveSurf(:,i); ContactPairs(4*i).SlaveIntegralPoint=4;
end
end
%  ===  Determine the contact state, then calculate contact stiffness and contact force   ===
function [ContactPairs,GKF,Residual]=DetermineContactState(FEMod,ContactPairs,...
    Dt,PreDisp,GKF,Residual,Disp)
IntegralPoint=[-0.57735026918963D0, -0.57735026918963D0; 0.57735026918963D0, -0.57735026918963D0;
    0.57735026918963D0,  0.57735026918963D0; -0.57735026918963D0, 0.57735026918963D0];
ContactPairs=ContactSearch(FEMod,ContactPairs,Disp,IntegralPoint);FricFac=FEMod.FricFac;
for i=1:size(ContactPairs,1)
    if(ContactPairs(i).CurMasterSurf(1)~=0)%Contact occur
        if(FricFac==0)||(ContactPairs(i).PreMasterSurf(1)==0)%First contact or Frictionless contact
            ContactPairs(i).CurContactState=2;%Slip contact state
            [GKF,Residual,ContactPairs]=CalculateContactKandF(FEMod,...
                ContactPairs,Dt,PreDisp,i,GKF,Residual,Disp,IntegralPoint);
        else
            CurIP=IntegralPoint(ContactPairs(i).SlaveIntegralPoint,:)';[Na,N1a,N2a]=GetSurfaceShapeFunction(CurIP(1),CurIP(2));
            [CurSlaveSurfXYZ,~]=GetSurfaceNodeLocation(FEMod, Disp, ContactPairs(i).SlaveSurf);
            Cur_x1=sum(Na.*CurSlaveSurfXYZ,1)';
            [Nb,~,~,]=GetSurfaceShapeFunction(ContactPairs(i).rp,ContactPairs(i).sp);
            [CurMasterSurfXYZ_p,~]=GetSurfaceNodeLocation(FEMod,Disp,ContactPairs(i).PreMasterSurf);
            Cur_x2_p=sum(Nb.*CurMasterSurfXYZ_p,1)';gs=Cur_x2_p-Cur_x1; tv=ContactPairs(i).pc*gs;
            Cur_N1Xa=sum(N1a.*CurSlaveSurfXYZ,1)';Cur_N2Xa=sum(N2a.*CurSlaveSurfXYZ,1)';
            Cur_n=cross(Cur_N1Xa,Cur_N2Xa); Cur_n=Cur_n./(sqrt(Cur_n'*Cur_n));
            tn_trial=abs(tv'*Cur_n); tt_trial=(norm(tv)*norm(tv)-tn_trial*tn_trial)^0.5;
            fai=tt_trial-FricFac*tn_trial;
            if(fai<0), ContactPairs(i).CurContactState=1;%Judging contact state: stick (1) or slip (2)
            else, ContactPairs(i).CurContactState=2; end
            [GKF,Residual,ContactPairs]=CalculateContactKandF(FEMod,...
            ContactPairs,Dt,PreDisp,i,GKF,Residual,Disp,IntegralPoint);
        end
    end
end
end
%  ===  Find nearest master surface corresponding to slave surface point  ===
function ContactPairs=ContactSearch(FEMod,ContactPairs,Disp,IntegralPoint)
for i=1:size(ContactPairs,1)
    Exist=0;
    [SlaveSurfNodeXYZ,~]=GetSurfaceNodeLocation(FEMod,Disp,ContactPairs(i).SlaveSurf);
    CurIP=IntegralPoint(ContactPairs(i).SlaveIntegralPoint,:)';
    [N,N1,N2]=GetSurfaceShapeFunction(CurIP(1),CurIP(2));SlavePoint=sum(N.*SlaveSurfNodeXYZ,1)';
    N1X=sum(N1.*SlaveSurfNodeXYZ,1)';N2X=sum(N2.*SlaveSurfNodeXYZ,1)';
    SlavePointTan=[N1X,N2X];
    %Obtain parameter coordinate of contact point
    [rr,ss,MasterEle,MasterSign,gg,Exist]=GetContactPointbyRayTracing(FEMod,Disp,SlavePoint,SlavePointTan);
    if(Exist==1)% Save contact pair information
        ContactPairs(i).CurMasterSurf(1)=MasterEle; ContactPairs(i).CurMasterSurf(2)=MasterSign;
        ContactPairs(i).rc=rr; ContactPairs(i).sc=ss;ContactPairs(i).Cur_g=gg;
    else
        ContactPairs(i).CurMasterSurf=[0;0];ContactPairs(i).rc=0; ContactPairs(i).sc=0;
        ContactPairs(i).Cur_g=0; ContactPairs(i).CurContactState=0;
    end
end
end
%  ===  Obtain master surface contact point by RayTracing ===
function [rr,ss,MasterEle,MasterSign,gg,Exist]=GetContactPointbyRayTracing(FEMod,Disp,SlavePoint,SlavePointTan)
Tol=1e-4; Exist=-1; MinDis=1e8; MinGrow=0; Ming=-1e3; MinMasterPoint=[];
AllMasterNode=zeros(size(FEMod.MasterSurf,2),4);
for i=1:size(FEMod.MasterSurf,2)%Find node closest to Integral point from slave surface
    [MasterSurfNode]=GetSurfaceNode(FEMod.Eles(FEMod.MasterSurf(1,i),:),FEMod.MasterSurf(2,i));
    AllMasterNode(i,:)=MasterSurfNode;MasterSurfDOF=zeros(size(MasterSurfNode,2)*3,1);
    for m=1:size(MasterSurfNode,2)
        MasterSurfDOF((3*m-2):3*m,1)=[(3*MasterSurfNode(m)-2):3*MasterSurfNode(m)]'; end
    MasterSurfDis=reshape(Disp(MasterSurfDOF),3,size(MasterSurfNode,2))';
    MasterSurfXYZ=FEMod.Nodes(MasterSurfNode,:)+MasterSurfDis;
    for j=1:4
        ll=MasterSurfXYZ(j,:)'-SlavePoint; Distance=sqrt(ll'*ll);
        if Distance<MinDis, MinDis=Distance; MinMasterPoint=MasterSurfNode(j); end
    end
end
[AllMinMasterSurfNum,~]=find(AllMasterNode==MinMasterPoint);%elements corresponding to nearest node
ContactCandidate=zeros(size(AllMinMasterSurfNum,1),8);ContactCandidate(:,5)=1e7;
for i=1:size(AllMinMasterSurfNum,1)
    MasterSurfNode=AllMasterNode(AllMinMasterSurfNum(i),:);MasterSurfDOF=zeros(size(MasterSurfNode,2)*3,1);
    for m=1:size(MasterSurfNode,2)
        MasterSurfDOF((3*m-2):3*m,1)=[(3*MasterSurfNode(m)-2):3*MasterSurfNode(m)]'; 
    end
    MasterSurfDis=reshape(Disp(MasterSurfDOF),3,size(MasterSurfNode,2))';
    MasterSurfXYZ=FEMod.Nodes(MasterSurfNode,:)+MasterSurfDis;
    r=0;s=0;
    for j=1:100000000%Parameter coordinates of RayTracing point are obtained by NR method
        [N,N1,N2]=GetSurfaceShapeFunction(r,s);
        NX=sum(N.*MasterSurfXYZ,1)';N1X=sum(N1.*MasterSurfXYZ,1)';N2X=sum(N2.*MasterSurfXYZ,1)';
        fai=[(SlavePoint-NX)'*SlavePointTan(:,1);(SlavePoint-NX)'*SlavePointTan(:,2)];
        if(j==500), r=100000; Exist=-1; break; end %no contact surface
        if max(abs(fai))<Tol, break; end
        k11=N1X'*SlavePointTan(:,1);k12=N2X'*SlavePointTan(:,1);
        k21=N1X'*SlavePointTan(:,2);k22=N2X'*SlavePointTan(:,2);
        KT=[k11 k12;k21 k22];drs=KT\fai; r=r+drs(1); s=s+drs(2);
    end
    if((abs(r)<=1.01)&&(abs(s)<1.01))%Save nearest RayTracing point parameter coordinates
        v=cross(SlavePointTan(:,1),SlavePointTan(:,2)); v=v./(sqrt(v'*v)); g=(NX-SlavePoint)'*v;
        ContactCandidate(i,1)=FEMod.MasterSurf(1,AllMinMasterSurfNum(i));
        ContactCandidate(i,2)=FEMod.MasterSurf(2,AllMinMasterSurfNum(i));
        ContactCandidate(i,3)=r; ContactCandidate(i,4)=s; ContactCandidate(i,5)=g; ContactCandidate(i,6:8)=v';
        if(Exist==-1||Exist==0)
            if(g>=0)&&(abs(Ming)>abs(g)), Exist=0; MinGrow=i; Ming=g;
            elseif(g<0), Exist=1; MinGrow=i; Ming=g; end
        elseif(Exist==1)
            if(g<0)&&(abs(Ming)>abs(g)&&g<0), Exist=1; MinGrow=i; Ming=g; end
        end
    end
end
if (Exist==0)%Exist master surface but no contact occurred
    MasterEle=ContactCandidate(MinGrow,1); MasterSign=ContactCandidate(MinGrow,2);
    rr=ContactCandidate(MinGrow,3); ss=ContactCandidate(MinGrow,4);
    gg=ContactCandidate(MinGrow,5);
elseif(Exist==1)%Exist master surface and contact occurred
    MasterEle=ContactCandidate(MinGrow,1); MasterSign=ContactCandidate(MinGrow,2);
    rr=ContactCandidate(MinGrow,3); ss=ContactCandidate(MinGrow,4);
    gg=ContactCandidate(MinGrow,5);
elseif(Exist==-1)%No master surface 
    MasterEle=1e10; MasterSign=1e10;rr=1e10; ss=1e10; gg=1e10;
end
end
%  ===  Obtain contact stiffness and contact force  ===
function [GKF,Residual,ContactPairs]=CalculateContactKandF(FEMod,ContactPairs,Dt,...
    PreDisp,i,GKF,Residual,Disp,IntegralPoint)
FricFac=FEMod.FricFac;
if(ContactPairs(i).CurContactState==1)%Stick contact state
    CurIP=IntegralPoint(ContactPairs(i).SlaveIntegralPoint,:)';[Na,N1a,N2a]=GetSurfaceShapeFunction(CurIP(1),CurIP(2));
    [CurSlaveSurfXYZ,SlaveSurfDOF]=GetSurfaceNodeLocation(FEMod, Disp, ContactPairs(i).SlaveSurf);
    Cur_x1=sum(Na.*CurSlaveSurfXYZ,1)';
    Cur_N1Xa=sum(N1a.*CurSlaveSurfXYZ,1)'; Cur_N2Xa=sum(N2a.*CurSlaveSurfXYZ,1)';
    Cur_n=cross(Cur_N1Xa,Cur_N2Xa);Cur_n=Cur_n./(sqrt(Cur_n'*Cur_n));
    J1=sqrt(cross(Cur_N1Xa,Cur_N2Xa)'*cross(Cur_N1Xa,Cur_N2Xa));
    [Nb,N1b,N2b]=GetSurfaceShapeFunction(ContactPairs(i).rp,ContactPairs(i).sp);
    [CurMasterSurfXYZ_rpsp,MasterSurfDOF]=GetSurfaceNodeLocation(FEMod,Disp,ContactPairs(i).PreMasterSurf);
    Cur_x2_p=sum(Nb.*CurMasterSurfXYZ_rpsp,1)';gs=Cur_x2_p-Cur_x1;tv=ContactPairs(i).pc*gs;
    ContactPairs(i).Pressure=abs(tv'*Cur_n);ContactPairs(i).Traction=abs(sqrt(tv'*tv));%save pressure
    ContactNodeForce=zeros(24,1);ContactDOF=[SlaveSurfDOF;MasterSurfDOF];
    for a=1:4
        ContactNodeForce((3*a-2):3*a,:)=Na(a)*J1*tv;ContactNodeForce(((3*a-2):3*a)+12,:)=-Nb(a)*J1*tv;
    end
    Residual(ContactDOF,:) = Residual(ContactDOF,:) +ContactNodeForce;
    Cur_g1_hat_slave=TransVect2SkewSym(Cur_N1Xa); Cur_g2_hat_slave=TransVect2SkewSym(Cur_N2Xa);
    Ac=(kron(N1a',Cur_g2_hat_slave)-kron(N2a',Cur_g1_hat_slave))/J1;
    Stick_K11=zeros(12);Stick_K12=zeros(12);Stick_K21=zeros(12);Stick_K22=zeros(12);
    for aa=1:4, for bb=1:4
            tempK=(-Na(aa)*Na(bb)*ContactPairs(i).pc*eye(3)-Na(aa)*(-tv*(Ac(:,(3*bb-2):3*bb)*Cur_n)'))*J1;
            Stick_K11((3*aa-2):3*aa,(3*bb-2):3*bb)=Stick_K11((3*aa-2):3*aa,(3*bb-2):3*bb)+tempK;
            tempK=(Na(aa)*Nb(bb)*ContactPairs(i).pc*eye(3))*J1;
            Stick_K12((3*aa-2):3*aa,(3*bb-2):3*bb)=Stick_K12((3*aa-2):3*aa,(3*bb-2):3*bb)+tempK;
            tempK=(Nb(aa)*Na(bb)*ContactPairs(i).pc*eye(3)+Nb(aa)*(-tv*(Ac(:,(3*bb-2):3*bb)*Cur_n)'))*J1;
            Stick_K21((3*aa-2):3*aa,(3*bb-2):3*bb)=Stick_K21((3*aa-2):3*aa,(3*bb-2):3*bb)+tempK;
            tempK=(-Nb(aa)*Nb(bb)*ContactPairs(i).pc*eye(3))*J1;
            Stick_K22((3*aa-2):3*aa,(3*bb-2):3*bb)=Stick_K22((3*aa-2):3*aa,(3*bb-2):3*bb)+tempK;
        end;end
    Stick_K=[Stick_K11,Stick_K12; Stick_K21,Stick_K22];
    GKF(ContactDOF,ContactDOF)=GKF(ContactDOF,ContactDOF)-Stick_K;
elseif(ContactPairs(i).CurContactState==2)%Slip contact state
    tn=ContactPairs(i).Cur_g*ContactPairs(i).pc;CurIP=IntegralPoint(ContactPairs(i).SlaveIntegralPoint,:)';
    [Na,N1a,N2a]=GetSurfaceShapeFunction(CurIP(1),CurIP(2));
    [CurSlaveSurfXYZ,SlaveSurfDOF]=GetSurfaceNodeLocation(FEMod, Disp, ContactPairs(i).SlaveSurf);
    [PreSlaveSurfNodeXYZ,~]=GetSurfaceNodeLocation(FEMod,PreDisp,ContactPairs(i).SlaveSurf);
    Cur_x1=sum(Na.*CurSlaveSurfXYZ,1)';Pre_x1=sum(Na.*PreSlaveSurfNodeXYZ,1)';dx1=Cur_x1-Pre_x1;
    Pre_N1Xa=sum(N1a.*PreSlaveSurfNodeXYZ,1)';Pre_N2Xa=sum(N2a.*PreSlaveSurfNodeXYZ,1)';
    Cur_N1Xa=sum(N1a.*CurSlaveSurfXYZ,1)';Cur_N2Xa=sum(N2a.*CurSlaveSurfXYZ,1)';
    Cur_n=cross(Cur_N1Xa,Cur_N2Xa);Cur_n=Cur_n./(sqrt(Cur_n'*Cur_n));
    J1=sqrt(cross(Cur_N1Xa,Cur_N2Xa)'*cross(Cur_N1Xa,Cur_N2Xa));PN=eye(3)-Cur_n*Cur_n';
    dg1_slave=Cur_N1Xa-Pre_N1Xa; dg2_slave=Cur_N2Xa-Pre_N2Xa;
    m1=cross(dg1_slave,Cur_N2Xa)+cross(Cur_N1Xa,dg2_slave);c1=PN*m1/J1;
    [Nb,N1b,N2b]=GetSurfaceShapeFunction(ContactPairs(i).rc,ContactPairs(i).sc);
    [CurMasterSurfNodeXYZ,MasterSurfDOF]=GetSurfaceNodeLocation(FEMod,Disp,ContactPairs(i).CurMasterSurf);
    [PreMasterSurfNodeXYZ,~]=GetSurfaceNodeLocation(FEMod,PreDisp,ContactPairs(i).CurMasterSurf);
    Cur_x2=sum(Nb.*CurMasterSurfNodeXYZ,1)';Pre_x2=sum(Nb.*PreMasterSurfNodeXYZ,1)';
    Cur_N1Xb=sum(N1b.*CurMasterSurfNodeXYZ,1)';Cur_N2Xb=sum(N2b.*CurMasterSurfNodeXYZ,1)';
    dx2=Cur_x2-Pre_x2;r1=ContactPairs(i).Cur_g*c1+dx1-dx2;vr=r1/Dt;s1_temp=PN*vr;
    if((s1_temp'*s1_temp)^0.5>1e-8), s1=s1_temp/((s1_temp'*s1_temp)^0.5);
    else, s1=zeros(3,1); dh=0; end
    ContactNodeForce=zeros(24,1);tv=tn*(Cur_n+FricFac*s1);temp_f1a=tv*J1;
    ContactPairs(i).Pressure=abs(tn);ContactPairs(i).Traction=abs(sqrt(tv'*tv));%save pressure
    for a=1:4
        f1a=Na(a)*temp_f1a;f2b=-Nb(a)*temp_f1a;
        ContactNodeForce((3*a-2):3*a,:)=f1a;ContactNodeForce(((3*a-2):3*a)+12,:)=f2b;
    end
    ContactDOF=[SlaveSurfDOF;MasterSurfDOF];
    Residual(ContactDOF,:) = Residual(ContactDOF,:)+ContactNodeForce;
    A_ab=[Cur_N1Xa'*Cur_N1Xb   Cur_N1Xa'*Cur_N2Xb;Cur_N2Xa'*Cur_N1Xb   Cur_N2Xa'*Cur_N2Xb;];a_ab=inv(A_ab);
    g1_bar_slave=a_ab(1,1)*Cur_N1Xb+a_ab(2,1)*Cur_N2Xb;g2_bar_slave=a_ab(1,2)*Cur_N1Xb+a_ab(2,2)*Cur_N2Xb;
    g1_bar_master=a_ab(1,1)*Cur_N1Xa+a_ab(1,2)*Cur_N2Xa;g2_bar_master=a_ab(2,1)*Cur_N1Xa+a_ab(2,2)*Cur_N2Xa;
    N1=Cur_n*Cur_n'; N1_bar=eye(3)-Cur_N1Xa*g1_bar_slave'-Cur_N2Xa*g2_bar_slave';
    mc1_bar=kron(N1a',g1_bar_slave)+kron(N2a',g2_bar_slave);
    mb2_bar=kron(N1b',g1_bar_master)+kron(N2b',g2_bar_master);
    Cur_g1_hat_slave=TransVect2SkewSym(Cur_N1Xa);Cur_g2_hat_slave=TransVect2SkewSym(Cur_N2Xa);
    Ac=(kron(N1a',Cur_g2_hat_slave)-kron(N2a',Cur_g1_hat_slave))/J1;N1_wave=Cur_n*(N1_bar*Cur_n)';
    Mc1_bar=[Cur_n*mc1_bar(:,1)'  Cur_n*mc1_bar(:,2)'  Cur_n*mc1_bar(:,3)'  Cur_n*mc1_bar(:,4)'];
    Mb2_bar=[-Cur_n*mb2_bar(:,1)'  -Cur_n*mb2_bar(:,2)'  -Cur_n*mb2_bar(:,3)'  -Cur_n*mb2_bar(:,4)'];
    N12a=[N1a';N2a'];N12b=[N1b';N2b'];Gbc=ContactPairs(i).Cur_g*[N12b'*a_ab*N12a];
    Frictionless_K11=zeros(12);Frictionless_K12=zeros(12);Frictionless_K21=zeros(12);Frictionless_K22=zeros(12);
    for aa=1:4  for bb=1:4
            tempK=(-Na(aa)*Na(bb)*ContactPairs(i).pc*N1_wave-Na(aa)*tn*(Ac(:,(3*bb-2):(3*bb))+Mc1_bar(:,(3*bb-2):(3*bb))*N1))*J1;
            Frictionless_K11((3*aa-2):3*aa,(3*bb-2):3*bb)=Frictionless_K11((3*aa-2):3*aa,(3*bb-2):3*bb)+tempK;
            tempK= (Na(aa)*Nb(bb)*ContactPairs(i).pc*N1_wave)*J1;
            Frictionless_K12((3*aa-2):3*aa,(3*bb-2):3*bb)=Frictionless_K12((3*aa-2):3*aa,(3*bb-2):3*bb)+tempK;
            tempK=(Nb(aa)*Na(bb)*ContactPairs(i).pc*N1_wave+Nb(aa)*tn*(Ac(:,(3*bb-2):(3*bb))+Mc1_bar(:,(3*bb-2):(3*bb))*N1)+...
                Na(bb)*tn*Mb2_bar(:,(3*aa-2):3*aa)+Gbc(aa,bb)*tn*N1)*J1;
            Frictionless_K21((3*aa-2):3*aa,(3*bb-2):3*bb)=Frictionless_K21((3*aa-2):3*aa,(3*bb-2):3*bb)+tempK;
            tempK=(-Nb(aa)*Nb(bb)*ContactPairs(i).pc*N1_wave-Nb(bb)*tn*Mb2_bar(:,(3*aa-2):3*aa))*J1;
            Frictionless_K22((3*aa-2):3*aa,(3*bb-2):3*bb)=Frictionless_K22((3*aa-2):3*aa,(3*bb-2):3*bb)+tempK;
        end;end
    FrictionlessK=[Frictionless_K11,Frictionless_K12;Frictionless_K21,Frictionless_K22];
    Frictional_K=zeros(24);
    if(FricFac~=0&&((s1'*s1)^0.5>1e-8))
        Q1=((Cur_n'*m1)*eye(3)+Cur_n*m1')/J1;
        dg1_hat_slave=TransVect2SkewSym(dg1_slave); dg2_hat_slave=TransVect2SkewSym(dg2_slave);
        Ac1_bar=(kron(N1a',dg2_hat_slave)-kron(N2a',dg1_hat_slave))/J1;
        dh=sqrt((PN*vr)'*(PN*vr))*Dt; Ps=(eye(3)-s1*s1')/dh;
        R1=((Cur_n'*r1)*eye(3)+Cur_n*r1')/ContactPairs(i).Cur_g;
        B1=(Ps*c1)*(N1_bar*Cur_n)'-Ps*PN;
        L1=ContactPairs(i).Cur_g*Ps*(PN*Q1+R1-eye(3))*PN;
        Jc1=L1*Ac-ContactPairs(i).Cur_g*Ps*PN*Ac1_bar;
        hc1_add=N1*mc1_bar+Ac*kron(eye(4),Cur_n);hc1_sub=N1*mc1_bar-Ac*kron(eye(4),Cur_n);
        S1=s1*Cur_n';S1_wave=s1*(N1_bar*Cur_n)';
        Frictional_K11=zeros(12);Frictional_K12=zeros(12);Frictional_K21=zeros(12);Frictional_K22=zeros(12);
        for aa=1:4, for bb=1:4
                tempK=(-Na(aa)*Na(bb)*FricFac*(ContactPairs(i).pc*S1_wave+tn*B1)-Na(aa)*FricFac*...
                    tn*(s1*hc1_sub(:,bb)'+ContactPairs(i).Cur_g*Ps*c1*hc1_add(:,bb)'-Jc1(:,(3*bb-2):3*bb)))*J1;
                Frictional_K11((3*aa-2):3*aa,(3*bb-2):3*bb)=Frictional_K11((3*aa-2):3*aa,(3*bb-2):3*bb)+tempK;
                tempK= (Na(aa)*Nb(bb)*FricFac*(ContactPairs(i).pc*S1_wave+tn*B1))*J1;
                Frictional_K12((3*aa-2):3*aa,(3*bb-2):3*bb)=Frictional_K12((3*aa-2):3*aa,(3*bb-2):3*bb)+tempK;
                tempK=(Nb(aa)*Na(bb)*FricFac*(ContactPairs(i).pc*S1_wave+tn*B1)+Nb(aa)*FricFac*tn*(s1*hc1_sub(:,bb)'+...
                    ContactPairs(i).Cur_g*Ps*c1*hc1_add(:,bb)'-Jc1(:,(3*bb-2):3*bb))+...
                Na(bb)*FricFac*tn*(-s1*mb2_bar(:,aa)')+Gbc(aa,bb)*FricFac*tn*S1)*J1;
                Frictional_K21((3*aa-2):3*aa,(3*bb-2):3*bb)=Frictional_K21((3*aa-2):3*aa,(3*bb-2):3*bb)+tempK;
                tempK=(-Nb(aa)*Nb(bb)*FricFac*(ContactPairs(i).pc*S1_wave+tn*B1)-Nb(bb)*FricFac*tn*(-s1*mb2_bar(:,aa)'))*J1;
                Frictional_K22((3*aa-2):3*aa,(3*bb-2):3*bb)=Frictional_K22((3*aa-2):3*aa,(3*bb-2):3*bb)+tempK;
            end;end
        Frictional_K=[Frictional_K11,Frictional_K12; Frictional_K21,Frictional_K22;];
    end
    ContactK=FrictionlessK+Frictional_K;
    GKF(ContactDOF,ContactDOF)=GKF(ContactDOF,ContactDOF)-ContactK;
end
end
%  ===  Shape function for 4-node surface  ===
function [N,N1,N2]=GetSurfaceShapeFunction(r,s)
N=[0.25*(r-1)*(s-1);-0.25*(r+1)*(s-1);0.25*(r+1)*(s+1);-0.25*(r-1)*(s+1)];
N1=[0.25*(s-1);-0.25*(s-1);0.25*(s+1);-0.25*(s+1)];N2=[0.25*(r-1);-0.25*(r+1);0.25*(r+1);-0.25*(r-1)];
end
%  ===  Obtain surface node location and surface node DOFs  ===
function [SurfNodeXYZ,SurfNodeDOF]=GetSurfaceNodeLocation(FEMod,Disp,Surf)
[SurfNode]=GetSurfaceNode(FEMod.Eles(Surf(1),:),Surf(2));SurfNodeDOF=zeros(size(SurfNode,2)*3,1);
for m=1:size(SurfNode,2), SurfNodeDOF((3*m-2):3*m,1)=[(3*SurfNode(m)-2):3*SurfNode(m)]'; end
SurfNodeDis=reshape(Disp(SurfNodeDOF),3,size(SurfNode,2))';
SurfNodeXYZ=FEMod.Nodes(SurfNode,:)+SurfNodeDis;
end
%  ===  Obtain skew-symmetric tensor of vector  ===
function [SkewSym]=TransVect2SkewSym(Vect)
SkewSym=[0,-Vect(3),Vect(2);Vect(3),0,-Vect(1);-Vect(2),Vect(1),0];
end
%  ===  Obtain surface node number  ===
function [SurfNode]=GetSurfaceNode(elementLE,SurfSign)
SurfNode=[];
if(SurfSign==1),SurfNode=elementLE(1,[4,3,2,1]);elseif(SurfSign==2),SurfNode=elementLE(1,[6,7,8,5]);
elseif(SurfSign==3),SurfNode=elementLE(1,[2,6,5,1]);elseif(SurfSign==4),SurfNode=elementLE(1,[2,3,7,6]);
elseif(SurfSign==5),SurfNode=elementLE(1,[3,4,8,7]);elseif(SurfSign==6),SurfNode=elementLE(1,[5,8,4,1]);
end
end
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