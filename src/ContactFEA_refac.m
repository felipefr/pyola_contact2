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
    Flag10 = 0; 
    Flag11 = 1;
    Flag20 = 1;
    ReductionNumber = 0;

    DispSave = Disp; 
    tempContactPairs=contactPairs;	
    Time0 = Time;

    Istep = Istep + 1; 
    Time = Time + Dt;

    while(Flag11 == 1)%Reduction loop
        NRConvergeNum=NRConvergeNum+1;  
        Flag11 = 0;

        %Check whether the calculation is completed
        % Todo: improve it
        if ((Time-1)>1E-10) 
            if ((1+Dt-Time)>1E-10),	
                Dt=1+Dt-Time; Time=1;
            else, 
                break; 
            end
        end

        Factor =Time;  
        SDisp = Dt*FEMod.Cons(:,3);  
        Iter = 0;
        PreDisp=Disp;

        while(Flag20 == 1) %NR loop
            Flag20 = 0; 
            Iter = Iter + 1;

            GKF = sparse(AllDOF,AllDOF); % can we choose sparse pattern? 
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
                ExtFVect(LOC) = ExtFVect(LOC) + Factor*FEMod.ExtF(:,3); 
            end
            Residual=Residual+ExtFVect;

            %Displacement boundary
            % Todo generalise to non-homogeneous
            if NCon~=0
                FixDOF=Dim*(FEMod.Cons(:,1)-1)+FEMod.Cons(:,2); 
                GKF(FixDOF,:)=zeros(NCon,AllDOF);
                GKF(FixDOF,FixDOF)=eye(NCon);
                Residual(FixDOF)=0;
                if Iter== 1, 
                    Residual(FixDOF) = SDisp(:); 
                end
            end

            if(Iter>1)
                FixDOF=Dim*(FEMod.Cons(:,1)-1)+FEMod.Cons(:,2); 
                FreeDOF=setdiff(1:AllDOF,FixDOF); % difference (set sense)

                [Resid,~]=max(abs(Residual(FreeDOF)));
                if Iter>2, 
                    fprintf(1,'%27d %14.5e \n',Iter,full(Resid));
                else, 
                    fprintf(1,'\n \t Time  Time step   Iter \t  Residual \n');
                    fprintf(1,'%10.5f %10.3e %5d %14.5e \n',Time,Dt,Iter,full(Resid)); 
                end
                
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
    norm(Disp)
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








