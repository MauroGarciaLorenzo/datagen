close all;
clearvars 
clc
path(pathdef)
addpath(genpath(pwd))

comb='1';


% TestDyn_4CIGs_OP_7_Y=readtable(['TestDyn_3CIGs_lines_OP_7_comb',comb,'_Y.xlsx']);
% TestDyn_2CIGs_OP_7_Y=readtable(['TestDyn_2CIGs_lines_OP_7_comb',comb,'_Y.xlsx']);
% 
% plot_Y_systems_comparison(TestDyn_2CIGs_OP_7_Y,TestDyn_4CIGs_OP_7_Y,4e3)
% 
% exportgraphics(gcf,['OP_7_comb',comb,'_maxfreq4e3.pdf'],'ContentType','vector')
% 
% plot_Y_systems_comparison(TestDyn_2CIGs_OP_7_Y,TestDyn_4CIGs_OP_7_Y,1e3)
% 
% exportgraphics(gcf,['OP_7_comb',comb,'.pdf'],'ContentType','vector')
% 
TestDyn_allcigs=readtable(['TestDyn_example_model_scenario_',comb,'_Y.xlsx']);
TestDyn_eqcigs=readtable(['TestDyn_example_model_scenario_',comb,'_eq_Y.xlsx']);

%plot_Y_systems_comparison(TestDyn_allcigs,TestDyn_eqcigs,4e3)

%exportgraphics(gcf,['TestDyn_example_model_scenario_',comb,'_maxfreq4e3.pdf'],'ContentType','vector')

plot_Y_systems_comparison(TestDyn_allcigs,TestDyn_eqcigs,1e3, 'Individual IBRs', 'Aggregated GFOR IBRs')

exportgraphics(gcf,['TestDyn_example_model_scenario_',comb,'.pdf'],'ContentType','vector')
