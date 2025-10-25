function Dtan = getIsotropicCelas(E, nu)
    fac = (E*(1-nu))/((1+nu)*(1-2*nu));
    Dtan = [1 nu/(1-nu) nu/(1-nu) 0 0 0;
            nu/(1-nu) 1 nu/(1-nu) 0 0 0;
            nu/(1-nu) nu/(1-nu) 1 0 0 0;
            0 0 0 (1-2*nu)/(2*(1-nu)) 0 0;
            0 0 0 0 (1-2*nu)/(2*(1-nu)) 0;
            0 0 0 0 0 (1-2*nu)/(2*(1-nu))];
    Dtan = fac*Dtan;
end