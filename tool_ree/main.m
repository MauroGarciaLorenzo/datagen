%% CLEAR WORKING ENVIRONMENT and SET PATHS

clear all 
close all
clc
clearvars -except data_PSCAD 
clc
path(pathdef)
addpath(genpath(pwd))

% path_pscad   = 'G:\.shortcut-targets-by-id\16EgEkoGAeU8JCz0SQG0Frr2kTcWjVjdk\PSCAD results Èlia\POD\TESTS\';

%% SELECT CASE

%excel        =   "01_data\IEEE_3bus_SG.xlsx";
tic
% %   118-bus thevenin
    %excel       = 'G:/Il mio Drive/Francesca 118 v2/Operation/tool_ree/tool_ree/01_data/IEEE_118bus_full_GFOLSLACK_6a.xlsx'
    excel        =   "01_data\IEEE_118bus_full_GFOLSLACK.xlsx";  %"01_data\IEEE_118bus\IEEE_118bus_01.xlsx"; %"01_data\IEEE_118bus\IEEE_118bus_02_1_025.xlsx";  %"01_data\IEEE_9bus\IEEE_9bus_FULL_03.xlsx";  %"01_data\IEEE_118bus\IEEE_118bus_6a_mod.xlsx";  
    linear       =  'IEEE_118bus_slack';  %'POD_nts_gfor'; %'IEEE_118bus_fullSG'; %'IEEE_118bus_fullGFOR'; 
    path_results = '01_data\IEEE_118bus\results\PSCC_tstep\';
    filePSCAD    =  'IEEE_118_SLACK_ZZ'; %'POD_full_GFOR_Q_POD_wo_negseq_9ms'; %'118_fullSG_loadchange_12'; %'118_alpha085beta025_Wnegseq';  %'118_fullGFOR_NEGSEQ_w_10_longer';  %'118_fullgfor_wo_negseq_w10_loadchange_12'; 

%   % 9-bus thevenin     
%     excel       =  "01_data\IEEE_9bus\PF_IEEE6bus.xlsx"; 
%     linear      =  'IEEE_6bus'; %'IEEE_6bus'; %'IEEE_2bus_lin';    
%     raw_file    = 'IEEE_9_no_thevenin.xlsx';
%     nonlinear   =  'IEEE9_bus_th'; %'IEEE2_bus';

  % 2-bus thevenin
%     excel       =  "01_data\IEEE_9bus\PF_2bus.xlsx";
%     linear      =  'IEEE_2bus_lin';    
%     raw_file    = 'IEEE_9_no_thevenin.xlsx';
%     nonlinear   =  'IEEE2_bus';
%     path_results = '01_data\IEEE_9bus\2_bus_rstep\';
%     filePSCAD   = '2bus_fullres_r_dis_resbreaker.txt';

% % 9-bus sg
%     excel        =  "01_data\IEEE_9bus\IEEE_9bus_sg_03.xlsx"; 
%     linear       =  'IEEE_9bus_sg_03_Rstep';    
%     path_results = '01_data\IEEE_9bus\9_bus_sg_03\results\';
%     filePSCAD    = 'ieee_sg_corrected.txt';

% % 9-bus GFOR
%     excel        =  "01_data\IEEE_9bus\IEEE_9bus_GFOR_03.xlsx"; 
%     linear       =  'IEEE_9bus_GFOR_03';    
%     path_results = '01_data\IEEE_9bus\9bus_GFOR_03\results\';
%     filePSCAD    = 'IEEE9_full_GFOR'; %'9_bus_full_GFOR_load_change';

% % 2-bus GFOR
%     excel        =  "01_data\IEEE_2bus\IEEE_2bus_GFOR_TH.xlsx"; 
%     linear       =  'IEEE_2bus_GFOR_TH';    %'IEEE_2bus_GFOR_PQ_CL';    
%     path_results = '01_data\IEEE_2bus\2_bus_GFOR\results\';
%     filePSCAD    = '2_bus_GFOR__wo_neg_seq_thevenin_trafo_bergeron_load_change';

% % % 2-bus GFOL
%     excel        =  "01_data\IEEE_2bus\IEEE_2bus_GFOL_TH.xlsx"; 
%     linear       =  'IEEE_2bus_GFOL_TH_onlyCL';   %'IEEE_2bus_GFOL_TH_onlyNET';  %'IEEE_2bus_GFOL_TH_outputPQ'; %'IEEE_2bus_GFOL_TH';    %'IEEE_2bus_GFOL_TH_onlyNET_wpll'; 
%     path_results = '01_data\IEEE_2bus\2_bus_GFOL\results\';
%     filePSCAD    = '2_bus_GFOL_sinPLL_currentcontrol';

% % 9-bus GFOL
%     excel        =  "01_data\IEEE_9bus\IEEE_9bus_GFOL_01.xlsx"; 
%     linear       =  'IEEE_9bus_GFOL_01';    
%     path_results = '01_data\IEEE_9bus\9_bus_GFOL_01\results\';
%     filePSCAD    = '9_bus_gfol_pure'; 

%9-bus FULL
%     excel        =  "01_data\IEEE_9bus\IEEE_9bus_FULL_03_tests.xlsx"; 
%     linear       =  'IEEE_9bus_FULL_03';    
%     path_results = '01_data\IEEE_9bus\9_bus_FULL_03\results\';
%     filePSCAD    = '9bus_alpha085_beta_025_wo_negseq_wPLLfilterr'; 

%% READ DATA
    
% Read USER raw data & Generate excel file in SS_TOOL format
    %run user_data.m

% Read excel file and generate tables of grid elements
    run read_data.m

% Enter USER data for power-flow
    run PF_results.m
    
%% USER ADDITIONAL CODE

run user_code.m

%% GENERATE STATE-SPACE MODEL

% Generate AC & DC NET
run generate_NET.m 
run user_ss.m

%% BUILD FULL SYSTEM STATE-SPACE MODEL

% % Display
%     run display_io.m  %display_io(l_blocks);
%     ss_sys = connect(l_blocks{:}, input, output);
% 
% % % Select
%     bus_in = 12;
%     user_out =  [12 19];
%     load_out = [12 14];
%     
%     run connect_user.m

input={'NET.Rld8'};
output={'NET.vn12d','NET.vn12q'};

ss_sys = connect(l_blocks{:}, input, output);

%% SMALL-SIGNAL ANALYSIS

T_EIG   = FEIG(ss_sys,[0.25 0.25 0.25],'o',true);

% Plot and save data
T_EIG   = FEIG(ss_sys,[0.25 0.25 0.25],'o',false);
    head(T_EIG)
%     writetable(T_EIG, [path_results 'CASE_1_085_025.xlsx'])
%     exportgraphics(gcf,[path_results '6b_EIG_mod2.emf'])

T_modal = FMODAL(ss_sys);

FMODAL_REDUCED_th(ss_sys,[1],0.1)

toc
    %exportgraphics(gcf,[path_results 'CASE_1_085_025_PF_02.emf'])
% 
% %     ylim([0 3000])
% %     xlim([-1000 500])
% %     set(gcf,'Position',[100 100 400 400])
% %     saveas(gcf,[path_results '6b_mod2.fig'])
% %     
% % %% LINEAR MODEL
% % 
% % % Case study
% % run user_linear.m
% % 
% % if ~isfile(['00_tool/Linear Model/' linear '.slx'])
% %     generate_linear(ss_sys,linear,tstep_lin,Tsim_lin,deltax) % create simulink linear model
% % else
% %     open(linear) % open already existing model
% % end
% % 
% % % Simulate
% % out_lin = sim(linear); 
% % 
% % %% VALIDATE LINEAR MODEL
% % 
% % run validate_ss_user_full.m
% % 
% % delta_slk = 0.453117201293350-pi/3+pi/2;
% % sqrt(vsg_q0^2+vsg_d0^2)
% % (atan(-vsg_d0/vsg_q0)+delta_slk)*180/pi
% % P=vsg_q0*isg_q0+vsg_d0*isg_d0
% % Q=vsg_q0*isg_d0-vsg_d0*isg_q0
% % %% NON-LINEAR
% % 
% % results.global = results.bus;
% % results.global.Vm = results.global.Vm;   
% % results.th = T_TH;
% % results.th.V = results.th.V*sqrt(2)/sqrt(3); 
% % % results.load = Traw.load;
% % % results.load.P = results.load.PL/T_global.Sb_MVA;
% % % results.load.Q = results.load.QL/T_global.Sb_MVA;
% % results.load = T_load;
% % 
% % % Initialization
% % run NET_initialization.m
% % %run user_initialization.m  % adapt this script
% % 
% % % Non-linear model simulation parameters
% % %Tsim = 2.5;
% % %tstep = 0.7;
% % %set_param([nonlinear '/MMC-Pac-1/Control Pac/Pac-ref'],'Time', num2str(tstep),'Before','1','After','1.05')
% % 
% % % Simulate
% % open(join([nonlinear '.slx']))
% % out_nolin = sim(nonlinear);  
% % 
% % %% SENSITIVITY
% % 
% % run pod_base.m