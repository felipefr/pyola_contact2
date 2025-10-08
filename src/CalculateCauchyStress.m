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