function [T_NET,T_trafo,T_load,T_TH] =  xb2lc(T_NET,T_trafo,T_load,T_TH,f) 

    T_NET.L   = T_NET.X/(2*pi*f);
    T_NET.C   = T_NET.B/(2*pi*f)/2;

    T_trafo.L = T_trafo.X/(2*pi*f);
    T_trafo.C = T_trafo.B/(2*pi*f);

    T_load.L  = T_load.X/(2*pi*f);

    T_TH.X = T_TH.L*(2*pi*f);
end


