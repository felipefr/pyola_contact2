
function [] = updateContact(contactPairs)
for im=1:size(contactPairs,1)%Update contact information
    if contactPairs(im).CurContactState==0%no contact
        contactPairs(im).PreMasterSurf=[0;0]; contactPairs(im).rp=0; contactPairs(im).sp=0;
        contactPairs(im).PreContactState=0; contactPairs(im).Pre_g=0;
        contactPairs(im).Pressure=0;contactPairs(im).Traction=0;
    else%slip or stick contact
        contactPairs(im).PreMasterSurf=contactPairs(im).CurMasterSurf;
        contactPairs(im).rp=contactPairs(im).rc; contactPairs(im).sp=contactPairs(im).sc;
        contactPairs(im).PreContactState=contactPairs(im).CurContactState;
        contactPairs(im).Pre_g=contactPairs(im).Cur_g;
    end
    contactPairs(im).rc=0; contactPairs(im).sc=0;
    contactPairs(im).Cur_g=0;contactPairs(im).CurMasterSurf=[0;0];
    contactPairs(im).CurContactState=0;
end
end
