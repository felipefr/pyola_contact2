%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% These MATLAB codes are written by Bin Wang, Wenjie Zuo 
% Email: zuowenjie@jlu.edu.cn
% It is an educational MATLAB implementation of contact problem
% This MATLAB code can be downloaded from https://www.researchgate.net/profile/Wenjie-Zuo-3 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  ===  Main function  ===
function ContactFEA_simple()
% solver params
Tmax = 0.1;
Nit = 3;
NNRmax=20; 
tolNR = 1e-7;
TimeList = linspace(0.0, Tmax, 10); 

% Mesh
[FEMod]= ModelInformation_Beam();

% Material
E=FEMod.Prop(1); nu=FEMod.Prop(2);
Dtan= getIsotropicCelas(E,nu);

contactPairs=InitializeContactPairs(FEMod);

[NodeNum, Dim] = size(FEMod.Nodes); 
AllDOF = Dim*NodeNum;
FixDOF=Dim*(FEMod.Cons(:,1)-1)+FEMod.Cons(:,2);
FreeDOF=setdiff(1:AllDOF,FixDOF); % difference (set sense)
Disp=zeros(AllDOF,1);

% Main loop
for i = 1 : (Nit-1)	
    Time = TimeList(i+1);
    Dt = TimeList(i+1) - TimeList(i);
    LoadFac = Time;
    SDisp = Dt*FEMod.Cons(:,3); 
    PreDisp=Disp;
    normRes = 9999.9;
    fprintf(1,'\n \t Time = %10.5f \n', Time);
    
    for k = 1: NNRmax
        GKF = sparse(AllDOF,AllDOF); % can we choose another sparse pattern? 
        Residual=zeros(AllDOF,1);
        ExtFVect=zeros(AllDOF,1); 
        NCon=size(FEMod.Cons,1);
        
        %Internal force and tangent stiffness matrix
        [Residual,GKF]=GetStiffnessAndForce(FEMod,Disp,Residual,GKF,Dtan);

        [contactPairs,GKF,Residual]=DetermineContactState(FEMod,...
             contactPairs,Dt,PreDisp,GKF,Residual,Disp);
        
        %Load boundary
        % node number: FEMod.ExtF(:,1) ?
        % dof number: FEMod.ExtF(:,2) ?
        % value : FEMod.ExtF(:,3) ?
        if size(FEMod.ExtF,1)>0, 
            LOC = Dim*(FEMod.ExtF(:,1)-1)+FEMod.ExtF(:,2);
            ExtFVect(LOC) = ExtFVect(LOC) + LoadFac*FEMod.ExtF(:,3); 
        end
        Residual=Residual+ExtFVect;

        %Displacement boundary
        GKF(FixDOF,:)=zeros(NCon,AllDOF);
        GKF(FixDOF,FixDOF)=eye(NCon);
        Residual(FixDOF)=0;
        if k == 1, 
            Residual(FixDOF) = SDisp(:);
        else,
            normRes = norm(Residual);
            fprintf(1,'%27d %14.5e \n', k, full(normRes));
        end

        %Determine whether it converges
        if(normRes<tolNR) 
            updateContact(contactPairs);
            % contactPairs.CurContactState
            break;
        end
        

        IncreDisp = GKF\Residual; 
        Disp = Disp + IncreDisp;
    end

    fprintf("norm displacement = %10.5f \n" , norm(Disp));
end

% Plotting

% Plot Displacement magnitude
UM=(Disp([1:3:size(Disp,1)]).^2+Disp([2:3:size(Disp,1)]).^2+Disp([3:3:size(Disp,1)]).^2).^0.5;
PlotStructuralContours(FEMod.Nodes,FEMod.Eles,Disp,UM);title("Umag");

% % Plot Von Mises
% [NodeCauchyMises,NodeCauchy]=CalculateCauchyStress(FEMod,Disp,Dtan);
% PlotStructuralContours(FEMod.Nodes,FEMod.Eles,Disp,NodeCauchyMises);title("CauchyMises")
% 
% % Plot Contact
% PlotContactPressureAndTraction(FEMod,contactPairs)
% 
% %Plot contours Displacement
% for i=1:3,PlotStructuralContours(FEMod.Nodes,FEMod.Eles,Disp,Disp([i:3:size(Disp,1)]));title("U"+num2str(i));end
% 
% % Plot contours stress
% for i=1:6,PlotStructuralContours(FEMod.Nodes,FEMod.Eles,Disp,NodeCauchy(i,:));title("CauchyStress"+num2str(i));end

end








