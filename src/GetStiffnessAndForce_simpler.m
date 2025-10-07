% =================== FEM Assembly Functions =======================
%  ===  Obtain global stiffness matrix and internal force vector ===
function [Resdual,GKF]=GetStiffnessAndForce(Nodes,Eles,Disp,Resdual,GKF,Dtan)
XG=[-0.57735026918963D0, 0.57735026918963D0];WGT=[1.00000000000000D0, 1.00000000000000D0];
for IE=1:size(Eles,1)
    Elxy=Nodes(Eles(IE,:),:);IDOF=zeros(1,24);
    for I=1:8,II=(I-1)*3+1;IDOF(II:II+2)=(Eles(IE,I)-1)*3+1:(Eles(IE,I)-1)*3+3;end
    EleDisp=Disp(IDOF);EleDisp=reshape(EleDisp,3,8);%Element dispacement
    for LX=1:2, for LY=1:2, for LZ=1:2
                E1=XG(LX); E2=XG(LY); E3=XG(LZ);
                [Shpd, Det] = GetShapeFunction([E1 E2 E3], Elxy);
                FAC=WGT(LX)*WGT(LY)*WGT(LZ)*Det;
                F=EleDisp*Shpd' + eye(3);
                Strain=0.5*(F'*F-eye(3));   % Calculate Lagrangian strain
                Strain=ten2voigt(Strain, 'strain');
                Stress=Dtan*Strain;%PK2stress
                [BN, BG] = getBmatrices(Shpd, F);
                Resdual(IDOF) = Resdual(IDOF) - FAC*BN'*Stress;%Assemble internal force vector
                SIG=voigt2ten(Stress, 'stress');
                SHEAD=zeros(9);SHEAD(1:3,1:3)=SIG;SHEAD(4:6,4:6)=SIG;SHEAD(7:9,7:9)=SIG;
                EKF = BN'*Dtan*BN + BG'*SHEAD*BG;
                GKF(IDOF,IDOF)=GKF(IDOF,IDOF)+FAC*EKF;%Assemble tangent stiffness matrix
            end; end; end
end
end