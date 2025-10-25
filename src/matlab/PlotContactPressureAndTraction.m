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