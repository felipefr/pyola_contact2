
function [BN, BG] = getBmatrices(Shpd, F)
BN=zeros(6,24);BG=zeros(9,24);
for I=1:8
    COL=(I-1)*3+1:(I-1)*3+3;
    BN(:,COL)=[Shpd(1,I)*F(1,1) Shpd(1,I)*F(2,1) Shpd(1,I)*F(3,1);
        Shpd(2,I)*F(1,2) Shpd(2,I)*F(2,2) Shpd(2,I)*F(3,2);
        Shpd(3,I)*F(1,3) Shpd(3,I)*F(2,3) Shpd(3,I)*F(3,3);
        Shpd(1,I)*F(1,2)+Shpd(2,I)*F(1,1) Shpd(1,I)*F(2,2)+Shpd(2,I)*F(2,1) Shpd(1,I)*F(3,2)+Shpd(2,I)*F(3,1);
        Shpd(2,I)*F(1,3)+Shpd(3,I)*F(1,2) Shpd(2,I)*F(2,3)+Shpd(3,I)*F(2,2) Shpd(2,I)*F(3,3)+Shpd(3,I)*F(3,2);
        Shpd(1,I)*F(1,3)+Shpd(3,I)*F(1,1) Shpd(1,I)*F(2,3)+Shpd(3,I)*F(2,1) Shpd(1,I)*F(3,3)+Shpd(3,I)*F(3,1)];
    BG(:,COL)=[Shpd(1,I) 0         0;Shpd(2,I) 0         0;Shpd(3,I) 0         0;
        0         Shpd(1,I) 0;0         Shpd(2,I) 0;0         Shpd(3,I) 0;
        0         0         Shpd(1,I);0         0         Shpd(2,I);0         0         Shpd(3,I)];
end
end