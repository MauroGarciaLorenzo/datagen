function [Connectivity_Matrix_rl,T_NET,T_TH_missing,rl_T_nodes] = add_TH(T_TH,Connectivity_Matrix_rl,Connectivity_Matrix_PI,T_NET,rl_T_nodes)
    nodeACmax   = size(Connectivity_Matrix_rl,1)+1;
    number      = max(T_NET.number)+1;
    index_table = size(rl_T_nodes,1)+1;
    missing     = [];

    for th = 1:1:size(T_TH,1)
        if sum(Connectivity_Matrix_rl(T_TH.bus(th),:))>0 && sum(Connectivity_Matrix_PI(T_TH.bus(th),:))==0
          Connectivity_Matrix_rl(T_TH.bus(th),nodeACmax) = 1;
          Connectivity_Matrix_rl(nodeACmax,T_TH.bus(th)) = 1;
          T_NET(end+1,:) = {number,T_TH.bus(th),nodeACmax,T_TH.R(th), T_TH.X(th), 0, T_TH.L(th), 0};
          rl_T_nodes{index_table,1} = nodeACmax;
          rl_T_nodes{index_table,2} = {join(['Additional TH',num2str(T_TH.number(th))])};
          index_table=index_table+1;
          nodeACmax                 = nodeACmax+1;
          number                    = number+1;
        else
          missing = [missing,th];
        end
    end

    T_TH_missing = T_TH(missing,:);
    
end