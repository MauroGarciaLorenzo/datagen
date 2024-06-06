function [Connectivity_Matrix,T_NET,T_trafo_missing] = add_trafo(T_trafo,Connectivity_Matrix,T_NET)
    missing = [];
    for tf = 1:1:size(T_trafo,1)
        if sum(Connectivity_Matrix(T_trafo.bus_from(tf),:))>0 || sum(Connectivity_Matrix(T_trafo.bus_to(tf),:))>0
            Connectivity_Matrix(T_trafo.bus_from(tf),T_trafo.bus_to(tf))=1;
            Connectivity_Matrix(T_trafo.bus_to(tf),T_trafo.bus_from(tf))=1;
            T_NET(end+1,:) = {T_trafo.number(tf),T_trafo.bus_from(tf),T_trafo.bus_to(tf),T_trafo.R(tf),T_trafo.X(tf),0,T_trafo.L(tf),T_trafo.C(tf)};
        else
            missing = [missing,tf];
        end
    end
    T_trafo_missing = T_trafo(missing,:);
end