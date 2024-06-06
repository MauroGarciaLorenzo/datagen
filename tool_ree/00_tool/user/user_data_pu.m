%% IEEE 9-bus system

Traw.global        = readtable(raw_file,'Sheet','global');        %global data table 
Traw.NET           = readtable(raw_file,'Sheet','AC-NET');        %AC-NET table
Traw.load          = readtable(raw_file,'Sheet','load');          %Load table
Traw.TH            = readtable(raw_file,'Sheet','TH');            %TH table
Traw.PF            = readtable(raw_file,'Sheet','Results PF');    %TH table

Vn_kV = Traw.global.Vb_kV;
Vb = (Vn_kV*1e3)*sqrt(2)/sqrt(3);
Sb = Traw.global.Sb_MVA*1e6;
Ib = (2/3)*Sb/Vb;
Zb = Vb/Ib;

nbus = height(Traw.PF);
Traw.global.f_Hz = 60;

% LOADS: get R,X from P,Q (powerflow)
Traw.load.R = zeros(height(Traw.load),1);
Traw.load.X = zeros(height(Traw.load),1);
for idx = 1:height(Traw.load)
    bus = Traw.load.I(idx);
    %Traw.load.R(idx) = (Traw.PF.VM(Traw.PF.No == bus)*Vb_kV*1e3)^2/(Traw.load.PL(idx)*1e6)/Zb;
    %Traw.load.X(idx) = (Traw.PF.VM(Traw.PF.No == bus)*Vb_kV*1e3)^2/(Traw.load.QL(idx)*1e6)/Zb;
    Traw.load.R(idx) = ((Vn_kV*1e3)^2/(Traw.load.PL(idx)*1e6))/Zb;
    Traw.load.X(idx) = ((Vn_kV*1e3)^2/(Traw.load.QL(idx)*1e6))/Zb;
end

% TH: get R,L in p.u. from R,L (real)
Traw.TH.R = zeros(height(Traw.TH),1);
Traw.TH.X = zeros(height(Traw.TH),1);
Traw.TH.L = zeros(height(Traw.TH),1);
Traw.TH.P = zeros(height(Traw.TH),1);
Traw.TH.Q = zeros(height(Traw.TH),1);
Traw.TH.type = ones(height(Traw.TH),1);
for idx = 1:height(Traw.TH)
    bus = Traw.TH.I(idx);
    Zb_th = (Traw.TH.BASKV(idx)*1e3)^2/(Sb);
    Traw.TH.R(idx) = Traw.TH.R_ohm(idx)/Zb_th;
    Traw.TH.X(idx) = Traw.TH.L_H(idx)*(2*pi*Traw.global.f_Hz)/Zb_th;
    Traw.TH.P(idx) = Traw.TH.PG(idx)/Sb; 
    Traw.TH.Q(idx) = Traw.TH.QG(idx)/Sb; 
    Traw.TH.V(idx) = Traw.PF.VM(Traw.PF.No == bus);
    if Traw.PF.VA(Traw.PF.No == bus) == 0
        Traw.TH.type(idx) = 0;
    end
end

%% Case #1 : Thevenin equivalents 

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
 
%% MATLAB MODEL VALUES
f=50;
Psc = [247.5e6 192e6 128e6];
XR = [100000 100000 100000];
R_calc = zeros(1,3);
X_calc = zeros(1,3);
L_calc = zeros(1,3);

for idx = 1:height(Traw.TH)
    bus     = Traw.TH.I(idx);
    Zb_th   = (Traw.TH.BASKV(idx)*1e3)^2/(Sb_MVA*1e6);

    X = (Traw.TH.BASKV(idx)*1e3)^2/Psc(idx);
    R = X/XR(idx);
    L = X/(2*pi*f);

    X_pu = X/Zb_th;
    R_pu = R/Zb_th;
    L_pu = L/Zb_th;

    R_calc(idx) = R_pu;
    X_calc(idx) = X_pu;
    L_calc(idx) = L_pu;

    %Traw.TH.R(idx) = R_pu;
    %Traw.TH.X(idx) = X_pu;
end