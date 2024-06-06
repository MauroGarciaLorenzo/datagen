%% Manually enter data

%     % Number, bus, Snom, Pg (in pu system), Qg (in pu system)
%     results.gen = [1	1	2.6 0.71641	0.27046;
%                    2	4	3.10 1.63	0.06654;
%                    3	6	2.8 0.85	-0.1086;];
%     
%     % Number, Vn, Vpu, th (degrees)
%     results_bus = [1,230.0000,1.02579,  -2.2168;
%                    2,230.0000,0.99563,  -3.9888;
%                    3,230.0000,1.01265,  -3.6874;
%                    4,230.0000,1.02577,   3.7197;
%                    5,230.0000,1.01588,   0.7275;
%                    6,230.0000,1.03235,   1.9667];
%     results.bus = array2table(results_bus,'VariableNames',{'bus','Vb','V','theta'});

%% Built-in power-flow (Fanals)

%    %Convert X,B columns to R,L
%    [T_NET,T_trafo,T_load,T_TH] = xb2lc(T_NET,T_trafo,T_load,T_TH,T_global.f_Hz);
%    % Generates the Connectivity Matrix and the Table of nodes for the AC grid:
%    [connect_mtx, connect_mtx_PI, connect_mtx_rl, T_nodes] = generate_general_connectivity_matrix(AC_grid{:});
% 
%    results = powerFlow(T_nodes, T_global, T_DC_NET, AC_grid{:});
%    results.bus = results.global;
% 
%    % write results to T_user
%    for idx = 1:height(T_user)
%        bus_user = T_user{idx,"bus"};
%        T_user.P(idx) = results.user{results.user.bus == bus_user,"P"};
%        T_user.Q(idx)  = results.user{results.user.bus == bus_user,"Q"};
%        T_user.V(idx)  = results.bus{results.bus.bus == bus_user,"Vm"};
%        T_user.delta(idx)  = results.bus{results.bus.bus == bus_user,"theta"};
%    end


%% Read from excel

     results.bus = readtable(excel,'Sheet','PF','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve'); 