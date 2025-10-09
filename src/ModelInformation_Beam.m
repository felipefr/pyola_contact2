function [FEMod] =ModelInformation_Beam
fName='Beam.inp';[Nodes, Eles] = ReadMesh( fName );Prop=[210000 0.3];%Obtain node and element 
ForceNode=[705:5:735];ExtF=zeros(size(ForceNode,2),3);% Load boundary
for i=1:size(ForceNode,2),ExtF(i,:)=[ForceNode(i),2,-4e4];end
ConNode=[ 1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,...
  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,...
  33,  34,  35, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748,...
 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764,...
 765, 766, 767, 768, 769, 770, 771];
Cons=zeros(size(ConNode,2)*3,3);%Displacement boundary
for i=1:size(ConNode,2)
    Cons(3*i-2,:)=[ConNode(i),1,0];Cons(3*i-1,:)=[ConNode(i),2,0];Cons(3*i,:)=[ConNode(i),3,0];
end
MasterSurf=[483:3:792;4*ones(1,size(483:3:792,2))];%Contact boundary
SlaveSurf=[1:4:477;6*ones(1,size([1:4:477],2))];
FricFac=0.1;
FEMod=struct('Nodes',Nodes,'Eles',Eles,'Prop',Prop,'ExtF',ExtF,'Cons',Cons,'MasterSurf',MasterSurf,'SlaveSurf',SlaveSurf,'FricFac',FricFac);
end
function [nodes, ele] = ReadMesh( fname )%Obtain node and element from ".inp" 
fid = fopen(fname,'rt');S = textscan(fid,'%s','Delimiter','\n'); S = S{1};
idxS = strfind(S, 'Node'); idx1 = find(not(cellfun(@isempty, idxS)));
idxS = strfind(S, 'Element');idx2 = find(not(cellfun(@isempty, idxS)));
idxS = strfind(S, 'Nset');idx3 = find(not(cellfun(@isempty, idxS)));
nodes = S(idx1(1)+1:idx2(1)-1); nodes = cell2mat(cellfun(@str2num,nodes,'UniformOutput',false)); 
elements = S(idx2+1:idx3(1)-1) ;ele = [cell2mat(cellfun(@str2num,elements,'UniformOutput',false))];
nodes=nodes(:,2:end);ele=ele(:,2:end);
end
