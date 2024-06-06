%% Create list to store subsystems blocks
l_blocks = {};

%% Convert X,B columns to R,L
[T_NET,T_trafo,T_load,T_TH] = xb2lc(T_NET,T_trafo,T_load,T_TH,T_global.f_Hz);

%% AC Grid:

% Generates the Connectivity Matrix and the Table of nodes for the AC grid:

[connect_mtx, connect_mtx_PI, connect_mtx_rl, T_nodes] = generate_general_connectivity_matrix(T_NET,T_trafo, T_load, T_TH, T_SG, T_STATCOM, T_VSC, T_MMC_Pac_GFll,T_MMC_Vdc_GFll, T_b2b, T_user);
%NET_graph = generate_NET_graph(connect_mtx);

%% RL NET Add aditional elements

%Add transformers:
[connect_mtx_rl,T_NET_wTf,T_trafo_missing] = add_trafo(T_trafo,connect_mtx_rl,T_NET);
rl_T_nodes = generate_specific_T_nodes_v2(connect_mtx_rl,T_nodes);

%Add Thevenins:  
[connect_mtx_rl,T_NET_wTf_wTh,T_TH_missing,rl_T_nodes] = add_TH(T_TH,connect_mtx_rl,connect_mtx_PI,T_NET_wTf,rl_T_nodes); 
rl_T_NET = get_specific_NET(connect_mtx_rl,T_NET_wTf_wTh);

%% RL NET:

% Generates a graph of the AC RL grid:
NET_rl_graph = generate_NET_graph(connect_mtx_rl);

%figure
    %plot(NET_rl_graph)
    %title('RL NET')

% PI T_nodes:
PI_T_nodes = generate_specific_T_nodes_v2(connect_mtx_PI,T_nodes);

% Generates the State-Space of the AC RL grid:
rl_NET = generate_general_rl_NET_v3(connect_mtx_rl, rl_T_nodes, PI_T_nodes, rl_T_NET, T_global);

if ~(isempty(rl_NET.SS))
    l_blocks{end+1} = rl_NET.SS;
end

%% PI NET:

% Generates a graph of the AC PI grid:
PI_T_NET     = get_specific_NET(connect_mtx_PI,T_NET);
%NET_PI_graph = generate_NET_graph(connect_mtx_PI);
%figure
    %plot(NET_PI_graph)
    %title('PI NET')

% Generates the State-Space of the AC PI grid:
PI_NET = generate_general_PI_NET(connect_mtx_PI, connect_mtx_rl,PI_T_nodes,T_NET,T_trafo_missing,T_global);

if ~(isempty(PI_NET))
    l_blocks{end+1} = PI_NET;
end

%% Trafos:

for tf=1:1:size(T_trafo_missing,1)
    trafo = build_trafo(T_trafo_missing.number(tf),T_trafo_missing.bus_from(tf),T_trafo_missing.bus_to(tf),T_trafo_missing.R(tf),T_trafo_missing.X(tf),T_global.f_Hz);
    l_trafo{tf} = trafo;
    l_blocks{end+1} = trafo;
end
clear tf trafo

%% TH:

for th=1:1:size(T_TH_missing,1)
    thevenin = build_TH(T_TH_missing.bus(th),T_TH_missing.number(th),T_global.f_Hz,T_TH_missing.R(th),T_TH_missing.L(th));
    l_Thevenin{th} = thevenin;
    l_blocks{end+1} = thevenin;
end
clear th thevenin

%% Loads:

for l=1:1:size(T_load,1)
    bus = T_load{l,"bus"};
    if sum(connect_mtx_PI(bus,:))
          if T_load.L(l) == 0
            %load = build_Load_in_PI_R(T_load.bus(l),T_load.number(l),T_load.R(l));
            load = build_Load_in_PI_R_addR(T_load.bus(l),T_load.number(l),T_load.R(l),results.bus(results.bus.bus==bus,:),delta_slk);
          else
              %load = build_Load_in_PI(T_load.bus(l),T_load.number(l),T_global.f_Hz,T_load.R(l),T_load.L(l));
              load = build_Load_in_PI_addR(T_load.bus(l),T_load.number(l),T_global.f_Hz,T_load.R(l),T_load.L(l),results.bus(results.bus.bus==bus,:),delta_slk);
          end
    else
        %load = build_Load_in_rl(T_load.bus(l),T_load.number(l),connect_mtx_rl,T_nodes,T_global.f_Hz,T_load.R(l),T_load.L(l));
        load = build_Load_in_rl_R_addR(T_load.bus(l),T_load.number(l),connect_mtx_rl,T_nodes,T_global.f_Hz,T_load.R(l),T_load.L(l),results.bus(results.bus.bus==bus,:),delta_slk);
    end
    l_load{l} = load;
    l_blocks{end+1} = load;
end
clear l bus load

%% DC Grid

if ~(isempty(T_DC_NET))
    %Generates the Connectivity Matrix and the table of nodes for the DC grid:
    [DC_connect_mtx, T_DC_nodes] = generate_DC_connectivity_matrix(T_DC_NET, T_MMC_Vdc_GFll,T_MMC_Pac_GFll);
    % Generates a graph of the DC grid:
    DC_NET_graph = generate_NET_graph(DC_connect_mtx);
    % Generates the State-Space of the DC grid:
    DC_NET = generate_DC_NET(DC_connect_mtx,T_DC_nodes,T_DC_NET,T_MMC_Pac_GFll,T_MMC_Vdc_GFll);

    l_blocks{end+1} = DC_NET;
end
