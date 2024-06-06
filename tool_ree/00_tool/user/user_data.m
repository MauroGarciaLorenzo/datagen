%% IEEE 9-bus system

Traw.global        = readtable(raw_file,'Sheet','global');        %global data table 
Traw.NET           = readtable(raw_file,'Sheet','AC-NET');        %AC-NET table
Traw.load          = readtable(raw_file,'Sheet','load');          %Load table
Traw.TH            = readtable(raw_file,'Sheet','TH');            %TH table
Traw.PF            = readtable(raw_file,'Sheet','PF');    %TH table

Vb_kV  = Traw.global.Vb_kV;
Sb_MVA = Traw.global.Sb_MVA;
Zb     = (Vb_kV*1e3)^2/(Sb_MVA*1e6);
nbus = height(Traw.PF);

%Traw.global.f_Hz = 50;

% LOADS: get R,X from P,Q (powerflow)
Traw.load.R = zeros(height(Traw.load),1);
Traw.load.X = zeros(height(Traw.load),1);
for idx = 1:height(Traw.load)
    %bus = Traw.load.I(idx);

    % calculation based on power-flow
    %Traw.load.R(idx) = (Traw.PF.VM(Traw.PF.No == bus)*Vb_kV*1e3)^2/(Traw.load.PL(idx)*1e6)/Zb; 
    %Traw.load.X(idx) = (Traw.PF.VM(Traw.PF.No == bus)*Vb_kV*1e3)^2/(Traw.load.QL(idx)*1e6)/Zb;

    % calculation based on rated conditions
    %Traw.load.R(idx) = ((Vb_kV*1e3)^2/(Traw.load.PL(idx)*1e6))/Zb;
    %Traw.load.X(idx) = ((Vb_kV*1e3)^2/(Traw.load.QL(idx)*1e6))/Zb;

    % calculation based on pu rated conditions from excel
    bus = Traw.load.bus(idx);    
    Traw.load.R(idx) = ((Vb_kV*1e3)^2/(Traw.load.P(idx)*Sb_MVA*1e6))/Zb;
    Traw.load.X(idx) = ((Vb_kV*1e3)^2/(Traw.load.Q(idx)*Sb_MVA*1e6))/Zb;
    
end
% calculation based on pu rated conditions from excel
writematrix(table2array(Traw.load(:,{'number','bus','R','X'})),excel,'Sheet','load','Range','A2');         


% TH: get R,L in p.u. from R,L (real)
Traw.TH.R = zeros(height(Traw.TH),1);
Traw.TH.X = zeros(height(Traw.TH),1);
Traw.TH.L = zeros(height(Traw.TH),1);
Traw.TH.P = zeros(height(Traw.TH),1);
Traw.TH.Q = zeros(height(Traw.TH),1);
Traw.TH.type = ones(height(Traw.TH),1);
for idx = 1:height(Traw.TH)
    bus = Traw.TH.I(idx);
    Zb_th = (Traw.TH.BASKV(idx)*1e3)^2/(Sb_MVA*1e6);
    Traw.TH.R(idx) = Traw.TH.R_ohm(idx)/Zb_th;
    Traw.TH.X(idx) = Traw.TH.L_H(idx)*(2*pi*Traw.global.f_Hz)/Zb_th;
    Traw.TH.P(idx) = Traw.TH.PG(idx)/Sb_MVA; 
    Traw.TH.Q(idx) = Traw.TH.QG(idx)/Sb_MVA; 
    Traw.TH.V(idx) = Traw.PF.VM(Traw.PF.No == bus);
    if Traw.PF.VA(Traw.PF.No == bus) == 0
        Traw.TH.type(idx) = 0;
    end
end

%% Case #1 : Thevenin equivalents in 9-bus

% Add trafos to thevenins
X14 = 0.0576;
X27 = 0.0625;
X39 = 0.0586;
Traw.TH{1,"X"} = Traw.TH{1,"X"} + X14; 
Traw.TH{2,"X"} = Traw.TH{2,"X"} + X27;  
Traw.TH{3,"X"} = Traw.TH{3,"X"} + X39; 

Traw.TH{:,"L"} = Traw.TH.X/(2*pi*Traw.global.f_Hz);



%% EXPORT DATA

% Export data to excel file
writematrix(table2array(Traw.global),excel,'Sheet','global','Range','A2');     %global data table 
writematrix(table2array(Traw.NET),excel,'Sheet','AC-NET','Range','A2');        %Net table
writematrix(table2array(Traw.load(:,{'No','I','R','X'})),excel,'Sheet','load','Range','A2');          %Load table
writematrix(table2array(Traw.TH(:,{'No','I','P','Q','V','type','R','L'})),excel,'Sheet','TH','Range','A2'); %TH table

% Export powerflow results to desired format
% results.bus    = table('Size', [nbus 3],'VariableTypes', ["double","double","double"],'VariableNames', ["bus","Vm","theta"]);
% results.bus{:,:} = table2array(Traw.PF);
% writetable(results.bus,excel,'Sheet','PF_9bus');     % open excel --> rename buses
 
%% Parallel lines in IEEE-118

% % LINES
% %detect parallel PI lines in IEEE-118 (fromraw_file to excel)
% % row_double = [];
% % for idx_row = 1:height(T_NET)-1
% %     if (T_NET.bus_from(idx_row) == T_NET.bus_from(idx_row+1)) && (T_NET.bus_to(idx_row) == T_NET.bus_to(idx_row+1)) 
% %         row_double(end+1) = idx_row+1;
% % 
% %     end
% % end

for idx = 1:height(Traw.NET)-1
    bus_from_01 = Traw.NET.I(idx);
    bus_to_01   = Traw.NET.J(idx);
    bus_from_02 = Traw.NET.I(idx+1);
    bus_to_02   = Traw.NET.J(idx+1);

    if (bus_from_01 == bus_from_02) && (bus_to_01 == bus_to_02)
        Z1   = Traw.NET.R(idx)   + 1i*Traw.NET.X(idx);
        Z2   = Traw.NET.R(idx+1) + 1i*Traw.NET.X(idx+1);
        Z_eq = 1/(1/Z1 + 1/Z2);
        B_eq = Traw.NET.B(idx) + Traw.NET.B(idx+1); 
        [idx bus_from_01 bus_to_01 real(Z_eq) imag(Z_eq) B_eq]
    end
end

%% Calculate loads in IEEE-118

% LOADS: get R,X from P,Q (powerflow) for IEEE-118
for idx = 1:height(T_load)
    bus = T_load.bus(idx);
    % calculation based on power-flow
    %Traw.load.R(idx) = (results.Vm(results.bus == bus)*Vb_kV*1e3)^2/(T_load.P(idx)*1e6))/Zb;
    %Traw.load.X(idx) = (results.Vm(results.bus == bus)*Vb_kV*1e3)^2/(T_load.Q(idx)*1e6))/Zb;
    % calculation based on rated conditions
    T_load.R(idx) = ((Vb_kV*1e3)^2/(T_load.P(idx)*Sb))/Zb;
    if T_load.Q(idx) == 0
        T_load.X(idx) = 0;
    else
        T_load.X(idx) = ((Vb_kV*1e3)^2/(T_load.Q(idx)*Sb))/Zb;
    end
end

writematrix(table2array(T_load(:,{'R','X'})),excel,'Sheet','load','Range','C2');          %Load table


%% ARRANGE BUS NUMBERING for IEEE-118
% Detect which nodes are missing in PSCAD
bus_list = 1:118;
bus_missing = [];

for idx_bus = 1:length(bus_list)
    bus = bus_list(idx_bus);
    if ~ismember(bus,T_NET.bus_from)
        if ~ismember(bus,T_NET.bus_to)
            if ~ismember(bus,T_load.bus)
                if ~ismember(bus,T_TH.bus)
                    if ~ismember(bus,T_trafo.bus_from)
                        if ~ismember(bus,T_trafo.bus_to)
                            if ~ismember(bus,T_user.bus)
                                bus_missing(end+1) = bus;
                            end
                        end
                    end
                end
            end
        end
    end
end

%% SET USER DATA

bus_list = T_user.bus;
bus_list_new = repmat(bus_list,3,1);
num_list_new = 1:length(bus_list_new);
num_list_new = num_list_new';

% Number
writematrix(num_list_new,excel,'Sheet','user','Range','A2');  
% Bus number
writematrix(bus_list_new,excel,'Sheet','user','Range','B2');  

% Element
elementList_new = [repmat("SG",length(bus_list),1); repmat("GFOR",length(bus_list),1); repmat("GFOL",length(bus_list),1)];
writematrix(elementList_new,excel,'Sheet','user','Range','H2');  

% Sb
Traw_th = readtable(raw_file,'Sheet','TH');        %global data table 
Sb_list = repmat(Traw_th.Var8(2:end),3,1);
writematrix(Sb_list,excel,'Sheet','user','Range','I2');  

% Vb
Traw_th = readtable(raw_file,'Sheet','TH');        %global data table 
Vb_list = repmat(Traw_th.Var6(2:end),3,1);
writematrix(Vb_list,excel,'Sheet','user','Range','J2');  