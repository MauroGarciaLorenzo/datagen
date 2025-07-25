clearvars 
clc
close all
path(pathdef)
addpath(genpath(pwd))

%%

% Case name as in Excel files
    caseName = './01_data/cases/NREL_seed16_standalone/IEEE118_FULL_case0'; 
    % caseName_2 = './01_data/cases/seed17_GFOL_RL_percgfor091_PyStable/IEEE118_FULL'; 

% Relative path to the Folder for storing results
    path_results = '02_results\'; 

% Set power-flow source (0: Excel, 1: Fanals, 2: MATACDC) 
    fanals = 2; 

% Flag to indicate if T_case should be used (REE)
    shared_power = 0;

%% READ GRID TOPOLOGY DATA
 
% Create excel files and simulink models names 
    run set_file_names.m

% Read excel file and generate tables of grid elements
    run read_data.m

    T_gen = readtable(excel,'Sheet','gen');        %global data table 

   %%
define_constants;
mpc = loadcase('case118');     % Load the 118-bus system
mpc.baseMVA = 500; 
mpc.gen = mpc.gen(2:end,:);
mpc.bus(1,BUS_TYPE) = 1;
idx_original_slack = find(mpc.bus(:,BUS_TYPE) == 3);
mpc.bus(idx_original_slack,BUS_TYPE) = 2;

bus_new_slack =  T_gen.bus(find(T_gen.type == 0));
mpc.bus(find(mpc.bus(:,BUS_I) == bus_new_slack(1)),BUS_TYPE) = 3;

for bus = mpc.bus(:, BUS_I)'  % Loop over bus numbers (transpose to iterate properly)
    row = find(mpc.bus(:, BUS_I) == bus);  % Find the row index in mpc.bus

    % Find the corresponding row in T_load
    match_idx = find(T_load.bus == bus);

    if isempty(match_idx)
        mpc.bus(row, PD) = 0;
        mpc.bus(row, QD) = 0;
    else
        mpc.bus(row, PD) = T_load.P(match_idx) * 100;  % Assuming T_load.P is in per unit
        mpc.bus(row, QD) = T_load.Q(match_idx) * 100;  % Assuming T_load.Q is in per unit
    end
    mpc.bus(bus, VM) = T_PF.Vm(bus);
    %mpc.bus(bus, VA) = T_PF.theta(bus)*180/pi();
    
end

for bus = mpc.gen(:, BUS_I)'  % Loop over bus numbers (transpose to iterate properly)
    row = find(mpc.gen(:, BUS_I) == bus);  % Find the row index in mpc.bus

    match_idx = find(T_gen.bus == bus);

    if isempty(match_idx)
        mpc.gen(row, PG) = 0;
        mpc.gen(row, QG) = 0;
    else
        mpc.gen(row, PG) = sum(T_gen.P(match_idx) * 100);  % Assuming T_load.P is in per unit
        mpc.gen(row, QG) = sum(T_gen.Q(match_idx) * 100);  % Assuming T_load.Q is in per unit
    end

    mpc.gen(row,VG)= T_PF.Vm(find(T_PF.bus == bus));
end

table_buses= mpc.bus;
table_gens= mpc.gen;

results = runpf(mpc);          % Run the power flow

mpc.bus(:,PD)=mpc.bus(:,PD)*1.8;
mpc.bus(:,QD)=mpc.bus(:,QD)*1.8;

mpc.gen(:,PG)=mpc.gen(:,PG)*1.8;
mpc.gen(:,QG)=mpc.gen(:,QG)*1.8;
results = runpf(mpc);          % Run the power flow
% 
