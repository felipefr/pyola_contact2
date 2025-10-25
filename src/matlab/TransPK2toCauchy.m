function Cauchy=TransPK2toCauchy(F, StressPK2)
PK=[StressPK2(1) StressPK2(4) StressPK2(6);StressPK2(4) StressPK2(2) StressPK2(5);StressPK2(6) StressPK2(5) StressPK2(3)];
DETF = det(F);ST = F*PK*F'/DETF; Cauchy=[ST(1,1) ST(2,2) ST(3,3) ST(1,2) ST(2,3) ST(1,3)]';
end