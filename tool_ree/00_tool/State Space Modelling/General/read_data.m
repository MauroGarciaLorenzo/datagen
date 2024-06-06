%% Reads all the data from the main excel file and generates tables with it

%   [T_global, T_NET, T_DC_NET, T_trafo, T_load, T_TH, T_SG, T_STATCOM, T_VSC, T_MMC_Pac_GFll, T_MMC_Vdc_GFll, T_b2b, T_custom]     

    T_global        = readtable(excel,'Sheet','global','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');        %global data table --> PF
    T_NET           = readtable(excel,'Sheet','AC-NET','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');        %Net table
    T_DC_NET        = readtable(excel,'Sheet','DC-NET','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');        %DC Net table
    T_trafo         = readtable(excel,'Sheet','trafo','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');         %Transformer table
    T_load          = readtable(excel,'Sheet','load','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');          %Load table
    T_TH            = readtable(excel,'Sheet','TH','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');            %TH table
    T_SG            = readtable(excel,'Sheet','SG','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');            %SG table
    T_STATCOM       = readtable(excel,'Sheet','STATCOM','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');       %STATCOM table
    T_VSC           = readtable(excel,'Sheet','VSC','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');           %VSC table
    T_MMC_Pac_GFll  = readtable(excel,'Sheet','MMC-Pac-GFll','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');  %MMC table
    T_MMC_Vdc_GFll  = readtable(excel,'Sheet','MMC-Vdc-GFll','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');  %MMC table
    T_b2b           = readtable(excel,'Sheet','b2b','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');           %b2b table
%    T_user          = readtable(excel,'Sheet','user');        %user-custom additional elements table --> PF
    T_user          = readtable(excel,'Sheet','user','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');        %user-custom additional elements table --> PF
   
    AC_grid = {T_NET,T_trafo, T_load, T_TH, T_SG, T_STATCOM, T_VSC, T_MMC_Pac_GFll,T_MMC_Vdc_GFll, T_b2b, T_user};