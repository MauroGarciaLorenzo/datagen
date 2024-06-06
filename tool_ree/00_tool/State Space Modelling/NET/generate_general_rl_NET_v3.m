function [NET] = generate_general_rl_NET_v3(connect_mtx_rl, rl_T_nodes, PI_T_nodes, rl_T_NET, T_global)

%     T_NET.L = T_NET.X/2/pi/T_global.f_Hz;
%     T_NET.C   = T_NET.B/(2*pi*T_global.f_Hz);
    
    if sum(connect_mtx_rl,'all')
    
        %Order the number of the rl lines:
        for line=1:1:size(rl_T_NET,1)
            if rl_T_NET.bus_from(line)>rl_T_NET.bus_to(line)
                bus_from = rl_T_NET.bus_from(line);
                bus_to = rl_T_NET.bus_to(line);
                rl_T_NET(line,:) = {rl_T_NET.Number(line),bus_to,bus_from,rl_T_NET.R(line),rl_T_NET.L(line),rl_T_NET.C(line),rl_T_NET.Length(line),rl_T_NET.NumberLines(line)};
            end
        end
        
        %Flag that checks if there is at least one internal node
        internal_node=false;
        
        inputs = [];
        outputs = [];
        
         for i=1:1:size(rl_T_nodes.Node,1)
             strings = rl_T_nodes{i,2:end};
             strings = rmmissing(strings);
             if ((isempty(strings) & not(ismember(i,PI_T_nodes.Node)))) | (contains(strings,"TH") & not(contains(strings,"Additional")) & not(ismember(i,PI_T_nodes.Node))) %Chech if there is something connected to the node
                  %if sum(Connectivity_Matrix(i,:))>0  %&& sum(Connectivity_Matrix_PI(i,:))==0
                     outputs = [outputs;{join(['NET','.vn',num2str(rl_T_nodes.Node(i)),'q'])};{join(['NET','.vn',num2str(rl_T_nodes.Node(i)),'d'])}];
        % %         elseif sum(Connectivity_Matrix(i,:))>0  && sum(Connectivity_Matrix_PI(i,:))>0
        % %             inputs = [inputs;{join(['NET','.vn',num2str(T_nodes.Node(i)),'q'])};{join(['NET','.vn',num2str(T_nodes.Node(i)),'d'])}];
                  %end
             else
                inputs = [inputs;{join(['NET','.vn',num2str(rl_T_nodes.Node(i)),'q'])};{join(['NET','.vn',num2str(rl_T_nodes.Node(i)),'d'])}];
             end
         end
        for i=1:1:size(rl_T_NET.bus_from,1)
            if rl_T_NET.B(i)==0
                if rl_T_NET.bus_from(i)>rl_T_NET.bus_to(i)
                    outputs = [outputs; {join(['NET','.iq_',num2str(rl_T_NET.bus_to(i)),'_',num2str(rl_T_NET.bus_from(i))])};{join(['NET','.id_',num2str(rl_T_NET.bus_to(i)),'_',num2str(rl_T_NET.bus_from(i))])}];
                else
                    outputs = [outputs; {join(['NET','.iq_',num2str(rl_T_NET.bus_from(i)),'_',num2str(rl_T_NET.bus_to(i))])};{join(['NET','.id_',num2str(rl_T_NET.bus_from(i)),'_',num2str(rl_T_NET.bus_to(i))])}];
                end
            end
        end
        
        ii=1;
        for i = 1:1:size(rl_T_NET.bus_from,1)
            NET.RL(i).R=rl_T_NET.R(i);
            NET.RL(i).L=rl_T_NET.L(i);
            NET.RL(i).bus_from=rl_T_NET.bus_from(i);
            NET.RL(i).bus_to=rl_T_NET.bus_to(i);
            ss_rl{i} = crea_ss_rl(rl_T_NET.R(i),rl_T_NET.L(i),rl_T_NET.bus_from(i),rl_T_NET.bus_to(i),T_global.f_Hz);
            NET.RL(i).SS=ss_rl{i};
            strings = rl_T_nodes{rl_T_nodes.Node==rl_T_NET.bus_to(i),2:end};
            strings = rmmissing(strings);
            if ((isempty(strings) & not(ismember(rl_T_nodes{rl_T_nodes.Node==rl_T_NET.bus_to(i),1},PI_T_nodes.Node)))) | (contains(strings,"TH") & not(contains(strings,"Additional")) & not(ismember(rl_T_nodes{rl_T_nodes.Node==rl_T_NET.bus_to(i),1},PI_T_nodes.Node)))  
                internal_node=true;
                %line_nodes = find(Connectivity_Matrix(T_NET.bus_to(i),:)==1);
                line_nodes = find(connect_mtx_rl(rl_T_NET.bus_to(i),:)>0);
                R= zeros(size(line_nodes,2));
                L= zeros(size(line_nodes,2));
                rows_menys = rl_T_NET.bus_to == rl_T_NET.bus_to(i);
                rows_mes = rl_T_NET.bus_from == rl_T_NET.bus_to(i);
                
                [ss_union_q{ii}, ss_union_d{ii}] = crea_union(rl_T_NET,rows_mes,rows_menys,rl_T_NET.bus_to(i));
                ii=ii+1;
                rl_T_nodes{rl_T_nodes.Node==rl_T_NET.bus_to(i),2:end} = '-'; %In order to not repeat an isolated node
            end
        end
        
        if internal_node==true
            NET.SS = connect(ss_rl{:},ss_union_q{:},ss_union_d{:},inputs,outputs);
        else
            NET.SS = connect(ss_rl{:},inputs,outputs);
        end
    
    elseif ~sum(connect_mtx_rl,'all')
        NET.SS = {};
    end
    

          
    function [ss_union_q, ss_union_d] = crea_union(system_table,rows_mes,rows_menys,Node)
    
        R_mes = system_table.R(rows_mes);
        L_mes = system_table.L(rows_mes);
        
        
        R_menys = system_table.R(rows_menys);
        L_menys = system_table.L(rows_menys);
        
        nodes_mes = system_table.bus_to(rows_mes);
        nodes_menys = system_table.bus_from(rows_menys);
        
        %QQQQQQQQQ!!!!!!!
        outputnames_q = {join(['NET','.vn',num2str(Node),'q'])};
        inputnames_q = [];
        for j=1:1:size(nodes_menys,1)
            if nodes_menys(j)<Node
                inputnames_q = [inputnames_q;{join(['NET','.iq_',num2str(nodes_menys(j)),'_',num2str(Node)])}];
            else
                inputnames_q = [inputnames_q;{join(['NET','.iq_',num2str(Node),'_',num2str(nodes_menys(j))])}];
            end
        end
        for j=1:1:size(nodes_mes,1)
            if nodes_mes(j)>Node
                inputnames_q = [inputnames_q;{join(['NET','.iq_',num2str(Node),'_',num2str(nodes_mes(j))])}];
            else
                inputnames_q = [inputnames_q;{join(['NET','.iq_',num2str(nodes_mes(j)),'_',num2str(Node)])}];
            end
        end
        
        for j=1:1:size(nodes_menys,1)
            inputnames_q = [inputnames_q;{join(['NET','.vn',num2str(nodes_menys(j)),'q'])}];
        end
        for j=1:1:size(nodes_mes,1)
            inputnames_q = [inputnames_q;{join(['NET','.vn',num2str(nodes_mes(j)),'q'])}];
        end
    
        sumatori = 0;
        D_q = zeros(1,size(inputnames_q,1));
        %Is
        for j=1:1:size(nodes_menys,1)
            D_q(1,j) = R_menys(j)/L_menys(j);
            sumatori = sumatori - 1/L_menys(j);
        end
        jj =1;
    
        for j=(size(nodes_menys,1)+1):1:((size(nodes_menys,1)+size(nodes_mes,1)))
            D_q(1,j) = -R_mes(jj)/L_mes(jj);
            sumatori = sumatori - 1/L_mes(jj);
            jj=jj+1;
        end
        
        %Vs
        jj=1;
        for j=((size(nodes_menys,1)+1+size(nodes_mes,1))):1:(2*size(nodes_menys,1)+size(nodes_mes,1))
            D_q(1,j) = -1/L_menys(jj);
            jj=jj+1;
        end
    
        jj =1;
        for j=(2*size(nodes_menys,1)+1+size(nodes_mes,1)):1:(2*size(nodes_menys,1)+2*size(nodes_mes,1))
            D_q(1,j) = -1/L_mes(jj);
            jj=jj+1;
        end
        D_q = (1/sumatori)*D_q;
        %DDDDDDDDDDDD!!!!!!!
        outputnames_d = {join(['NET','.vn',num2str(Node),'d'])};
        inputnames_d = [];
        for j=1:1:size(nodes_menys,1)
            if nodes_menys(j)<Node
                inputnames_d= [inputnames_d;{join(['NET','.id_',num2str(nodes_menys(j)),'_',num2str(Node)])}];
            else
                inputnames_d= [inputnames_d;{join(['NET','.id_',num2str(Node),'_',num2str(nodes_menys(j))])}];
            end
        end
        for j=1:1:size(nodes_mes,1)
            if nodes_mes(j)>Node
                inputnames_d = [inputnames_d;{join(['NET','.id_',num2str(Node),'_',num2str(nodes_mes(j))])}];
            else
                inputnames_d = [inputnames_d;{join(['NET','.id_',num2str(nodes_mes(j)),'_',num2str(Node)])}];
            end
        end
        
        for j=1:1:size(nodes_menys,1)
            inputnames_d = [inputnames_d;{join(['NET','.vn',num2str(nodes_menys(j)),'d'])}];
        end
        for j=1:1:size(nodes_mes,1)
            inputnames_d = [inputnames_d;{join(['NET','.vn',num2str(nodes_mes(j)),'d'])}];
        end
    
        sumatori=0;
        D_d = zeros(1,size(inputnames_d,1));
        %Is
        for j=1:1:size(nodes_menys,1)
            D_d(1,j) = R_menys(j)/L_menys(j);
            sumatori = sumatori - 1/L_menys(j);
        end
        jj =1;
        for j=(size(nodes_menys,1)+1):1:((size(nodes_menys,1)+size(nodes_mes,1)))
            D_d(1,j) = -R_mes(jj)/L_mes(jj);
            sumatori = sumatori - 1/L_mes(jj);
            jj=jj+1;
        end
        
        %Vs
        jj=1;
        for j=((size(nodes_menys,1)+1+size(nodes_mes,1))):1:(2*size(nodes_menys,1)+size(nodes_mes,1))
            D_d(1,j) = -1/L_menys(jj);
            jj=jj+1;
        end
    
        jj =1;
        for j=(2*size(nodes_menys,1)+1+size(nodes_mes,1)):1:(2*size(nodes_menys,1)+2*size(nodes_mes,1))
            D_d(1,j) = -1/L_mes(jj);
            jj=jj+1;
        end
        D_d = (1/sumatori)*D_d;
        A = [0];
        B = zeros(1,size(D_q,2));
        C = [0];
        ss_union_q = ss(A,B,C,D_q,'statename','','inputname',inputnames_q,'outputname',outputnames_q);
        ss_union_d = ss(A,B,C,D_d,'statename','','inputname',inputnames_d,'outputname',outputnames_d);
    end
         
    function ss_rl = crea_ss_rl(R1,L1,bus_from,bus_to,f)
    
        A = [-R1/L1 -2*pi*f; 2*pi*f -R1/L1];
        B = [1/L1 0 -1/L1 0; 0 1/L1 0 -1/L1];
        C = [1 0; 0 1];
        D = [0 0 0 0; 0 0 0 0];
    
        if bus_from>bus_to
                inputname  = [{join(['NET','.vn',num2str(bus_to),'q']);join(['NET','.vn',num2str(bus_to),'d']);...
                              join(['NET','.vn',num2str(bus_from),'q']);join(['NET','.vn',num2str(bus_from),'d'])}];
                outputname = [{join(['NET','.iq_',num2str(bus_to),'_',num2str(bus_from)])};{join(['NET','.id_',num2str(bus_to),'_',num2str(bus_from)])}];
        else
                inputname = [{join(['NET','.vn',num2str(bus_from),'q']);join(['NET','.vn',num2str(bus_from),'d']);...
                              join(['NET','.vn',num2str(bus_to),'q']);join(['NET','.vn',num2str(bus_to),'d'])}];
                outputname = [{join(['NET','.iq_',num2str(bus_from),'_',num2str(bus_to)])};{join(['NET','.id_',num2str(bus_from),'_',num2str(bus_to)])}];
        end
    
    
        ss_rl = ss(A,B,C,D,'StateName',{join(['NET','.iq_',num2str(bus_from),'_',num2str(bus_to)]) ; join(['NET','.id_',num2str(bus_from),'_',num2str(bus_to)])},...
            'inputname',inputname,'outputname',outputname);
        
    %     ss_rl = ss(A,B,C,D,'StateName',{join(['NET','.iq_',num2str(bus_from),'_',num2str(bus_to)]) ; join(['NET','.id_',num2str(bus_from),'_',num2str(bus_to)])},...
    %         'inputname',{join(['NET','.vn',num2str(bus_from),'q']);join(['NET','.vn',num2str(bus_from),'d']);...
    %                      join(['NET','.vn',num2str(bus_to),'q']);join(['NET','.vn',num2str(bus_to),'d'])},...
    %         'outputname',{join(['NET','.iq_',num2str(bus_from),'_',num2str(bus_to)]) ; join(['NET','.id_',num2str(bus_from),'_',num2str(bus_to)])});
    end

end