%% Reads all the data from the main excel file and generates tables with it

%   [T_global, T_NET, T_DC_NET, T_trafo, T_load, T_TH, T_SG, T_STATCOM, T_VSC, T_MMC_Pac_GFll, T_MMC_Vdc_GFll, T_b2b, T_custom]

##    T_global = readtable(excel, 'Sheet', 'global', 'ReadRowNames', false, 'HeaderLines', 0, 'ReadVariableNames', true, 'PreserveVariableNames', true);
##    T_NET           = readtable(excel,'Sheet','AC-NET','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');        %Net table
##    T_DC_NET        = readtable(excel,'Sheet','DC-NET','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');        %DC Net table
##    T_trafo         = readtable(excel,'Sheet','trafo','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');         %Transformer table
##    T_load          = readtable(excel,'Sheet','load','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');          %Load table
##    T_TH            = readtable(excel,'Sheet','TH','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');            %TH table
##    T_SG            = readtable(excel,'Sheet','SG','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');            %SG table
##    T_STATCOM       = readtable(excel,'Sheet','STATCOM','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');       %STATCOM table
##    T_VSC           = readtable(excel,'Sheet','VSC','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');           %VSC table
##    T_MMC_Pac_GFll  = readtable(excel,'Sheet','MMC-Pac-GFll','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');  %MMC table
##    T_MMC_Vdc_GFll  = readtable(excel,'Sheet','MMC-Vdc-GFll','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');  %MMC table
##    T_b2b           = readtable(excel,'Sheet','b2b','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');           %b2b table
##%    T_user          = readtable(excel,'Sheet','user');        %user-custom additional elements table --> PF
##    T_user          = readtable(excel,'Sheet','user','ReadRowNames',false, 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');        %user-custom additional elements table --> PF
##
##    AC_grid = {T_NET,T_trafo, T_load, T_TH, T_SG, T_STATCOM, T_VSC, T_MMC_Pac_GFll,T_MMC_Vdc_GFll, T_b2b, T_user};


# Load the io package if not already loaded
pkg load io

# Function to read data from an Excel sheet and create a struct
function T = read_sheet_to_struct(excel, sheet_name)
  [~, ~, raw] = xlsread(excel, sheet_name);
  headers = raw(1, :);
  data = raw(2:end, :);
  T = struct();
  for col = 1:length(headers)
    header = headers{col};
    header = strrep(header, ' ', '_');  # Replace spaces with underscores for valid field names
    T.(header) = data(:, col);
  end
end

# Read all the sheets into structures
T_global = read_sheet_to_struct(excel, 'global');
T_NET = read_sheet_to_struct(excel, 'AC-NET');
T_DC_NET = read_sheet_to_struct(excel, 'DC-NET');
T_trafo = read_sheet_to_struct(excel, 'trafo');
T_load = read_sheet_to_struct(excel, 'load');
T_TH = read_sheet_to_struct(excel, 'TH');
T_SG = read_sheet_to_struct(excel, 'SG');
T_STATCOM = read_sheet_to_struct(excel, 'STATCOM');
T_VSC = read_sheet_to_struct(excel, 'VSC');
T_MMC_Pac_GFll = read_sheet_to_struct(excel, 'MMC-Pac-GFll');
T_MMC_Vdc_GFll = read_sheet_to_struct(excel, 'MMC-Vdc-GFll');
T_b2b = read_sheet_to_struct(excel, 'b2b');
T_user = read_sheet_to_struct(excel, 'user');

# Combine into a cell array
AC_grid = {T_NET, T_trafo, T_load, T_TH, T_SG, T_STATCOM, T_VSC, T_MMC_Pac_GFll, T_MMC_Vdc_GFll, T_b2b, T_user};


### Load the io package if not already loaded
##pkg load io
##
### Function to read data from an Excel sheet and create a table-like object
##function T = read_excel_as_table(excel, sheet_name)
##  [~, ~, raw] = xlsread(excel, sheet_name);
##  headers = raw(1, :);
##  data = raw(2:end, :);
##  T = table(data, headers);
##end
##
### Reading different sheets into table-like objects
##T_global = read_excel_as_table(excel, 'global');
##T_NET = read_excel_as_table(excel, 'AC-NET');
##T_DC_NET = read_excel_as_table(excel, 'DC-NET');
##T_trafo = read_excel_as_table(excel, 'trafo');
##T_load = read_excel_as_table(excel, 'load');
##T_TH = read_excel_as_table(excel, 'TH');
##T_SG = read_excel_as_table(excel, 'SG');
##T_STATCOM = read_excel_as_table(excel, 'STATCOM');
##T_VSC = read_excel_as_table(excel, 'VSC');
##T_MMC_Pac_GFll = read_excel_as_table(excel, 'MMC-Pac-GFll');
##T_MMC_Vdc_GFll = read_excel_as_table(excel, 'MMC-Vdc-GFll');
##T_b2b = read_excel_as_table(excel, 'b2b');
##T_user = read_excel_as_table(excel, 'user');
##
### Combining into a cell array
##AC_grid = {T_NET, T_trafo, T_load, T_TH, T_SG, T_STATCOM, T_VSC, T_MMC_Pac_GFll, T_MMC_Vdc_GFll, T_b2b, T_user};

