close all;
clearvars 
clc
path(pathdef)
addpath(genpath(pwd))

comb='3';

TestDyn_4CIGs_OP_7_Y=readtable(['TestDyn_3CIGs_lines_OP_7_comb',comb,'_Y.xlsx']);
TestDyn_2CIGs_OP_7_Y=readtable(['TestDyn_2CIGs_lines_OP_7_comb',comb,'_Y.xlsx']);

plot_Y_systems_comparison(TestDyn_2CIGs_OP_7_Y,TestDyn_4CIGs_OP_7_Y,4e3)

exportgraphics(gcf,['OP_7_comb',comb,'_maxfreq4e3.pdf'],'ContentType','vector')

plot_Y_systems_comparison(TestDyn_2CIGs_OP_7_Y,TestDyn_4CIGs_OP_7_Y,1e3)

exportgraphics(gcf,['OP_7_comb',comb,'.pdf'],'ContentType','vector')
