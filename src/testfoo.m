function [z]=testfoo(x,a)
    a = a + 2;
    b = a + x;
    z = [a,b];
end 