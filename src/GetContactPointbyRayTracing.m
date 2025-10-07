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