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