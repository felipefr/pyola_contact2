
%  ===  Initialize contact pair information  ===
function [B]=InitializeFakeStruct(d)
A.x = d;
A.y = 2*d;
B = [ A, A ]
% ContactPair.pc=1e6; ContactPair.SlaveSurf=[0;0];ContactPair.SlaveIntegralPoint=0;
% ContactPair.CurMasterSurf=[0;0];ContactPair.rc=0;ContactPair.sc=0;
% ContactPair.Cur_g=0;ContactPair.Pre_g=0;ContactPair.PreMasterSurf=[0;0];
% ContactPair.rp=0;ContactPair.sp=0;ContactPair.CurContactState=0;ContactPair.PreContactState=0;
% ContactPair.Pressure=0;ContactPair.Traction=0;
% ContactPairs = repmat(ContactPair, size(FEMod.SlaveSurf, 2) * 4, 1);
% for i=1:size(FEMod.SlaveSurf,2)
%     ContactPairs(4*i-3).SlaveSurf=FEMod.SlaveSurf(:,i); ContactPairs(4*i-3).SlaveIntegralPoint=1;
%     ContactPairs(4*i-2).SlaveSurf=FEMod.SlaveSurf(:,i); ContactPairs(4*i-2).SlaveIntegralPoint=2;
%     ContactPairs(4*i-1).SlaveSurf=FEMod.SlaveSurf(:,i); ContactPairs(4*i-1).SlaveIntegralPoint=3;
%     ContactPairs(4*i).SlaveSurf=FEMod.SlaveSurf(:,i); ContactPairs(4*i).SlaveIntegralPoint=4;
% end
end
