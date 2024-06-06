%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Grid-forming 2L-VSC (AC Grid)
% Date: 10/07/2020
% Model: S. Dadjo
% =========================================================================
% clear all; 
clc
format long;
% =========================================================================
%% Base values
Sb_n1=2.75e6;
Vb_n1=0.69e3;
Ib_n1=Sb_n1/Vb_n1;
Zb_n1=Vb_n1^2/Sb_n1;
Vb_dc_n1=2e3;
fb_n1=50;
wb_n1=2*pi*fb_n1;
Vpeak_n1=Vb_n1/sqrt(3)*sqrt(2);    
Ipeak_n1=Ib_n1/sqrt(3)*sqrt(2);

%% AC grid 1: Thevenin equivalent 

% only Z1 SCR=0.5
SCR1=0.5;                 % Short-circuit ratio
Scc1=SCR1*Sb_n1;          % Short-circuit power
Xcc1=Vb_n1^2/Scc1;        % Short-circuit impedance
Lcc1=Xcc1/(2*pi*fb_n1);    % Shot-circuit inductance
XR_ratio=3;
Rg_n1=sqrt(Xcc1^2/(XR_ratio^2+1));            % Thevenin resistance
Rg_n1=Rg_n1/Zb_n1;

Xg_n1=Rg_n1*XR_ratio;             
Lg_n1=Xg_n1/wb_n1;             % Thevenin inductance

% Z1 and Z2 (parallel) equivalent SCR=3
SCR2=3;                
Scc2=SCR2*Sb_n1;          
Xcct=Vb_n1^2/Scc2;        
Rg_t=sqrt(Xcct^2/(XR_ratio^2+1));
Rg_t=Rg_t/Zb_n1;
Xg_t=Rg_t*XR_ratio;             

Zg_n2=1/(1/complex(Rg_t,Xg_t)-1/complex(Rg_n1,Xg_n1));

Rg_n2=real(Zg_n2);            % Thevenin resistance
Lg_n2=imag(Zg_n2)/wb_n1;             % Thevenin inductance


% DC grid: ideal
Vdcref_n1=Vb_dc_n1;   

% Other parameters
sat_current=1.15;        % Current saturation
epsilon=1;              % Used for the current saturation
v_ini=1e-5;             % Initialization voltages
magnitud_delay=1e-5;    % Constant delay for simulation    
plots_step=10e-6;        % Plot samples step 

%% 2L-VSC 1 grid forming

Rc_n1 = 0.005; %0.005*Xn_n1/Zb_n1                  % Converter grid coupling filter resistance
Lc_n1 = 0.15/wb_n1; %0.1/wb_n1;   %0.1*Xc_n1/wb_n1       % Converter grid coupling filter inductance
Rseries_n1=5e-4/Zb_n1;                     %series resistance with source

Cac_n1=0.15/wb_n1;%0.15/Xn_n1/wb_n1;     % Converter grid coupling filter capacitance
% Rac_n1=1/(3*10*wb_n1*(0.15/Zb_n1/wb_n1))/Zb_n1;     % passive damping 
Rac_n1=1/(3*10*wb_n1*Cac_n1);     % passive damping
vaini_n1=1;                  % PCC initial voltages
vbini_n1=-1/2*1;
vcini_n1=-1/2*1;

P_ref_n1=0.5;
Q_ref_n1=0.1;

% Current control
taus=1e-3;
% kp_s_n1=(Lc_n1*Zb_n1)/taus*(Ib_n1/Vb_n1);
% ki_s_n1=(Rc_n1*Zb_n1)/taus*(Ib_n1/Vb_n1);
kp_s_n1=Lc_n1/taus;
ki_s_n1=Rc_n1/taus;

% AC voltage tuning: 
kp_vac_n1=4.789*(Vb_n1/Ib_n1);
ki_vac_n1=42.05*(Vb_n1/Ib_n1);
% kp_vac_n1=4.789*Ib_n1/Vb_n1;
% ki_vac_n1=42.05*Ib_n1/Vb_n1;

% feedforward filters
tau_u=0.1e-3;
tau_ig=0.1e-3;

% Droop parameters
k_droop_f_n1=0.05; %pu 0.05*w_n1/Sn_n1
tau_droop_f=1/50;
k_droop_u_n1=(0.02/0.3)*(sqrt(2)/sqrt(3)); %pu Vpeak_n1/Sn_n1
tau_droop_u=1/50;

% PLL tuning NOT USED
ts_pll_n1=0.025;
xi_pll_n1=0.707;
omega_pll_n1=4/(ts_pll_n1*xi_pll_n1);
kp_pll1=xi_pll_n1*2*omega_pll_n1/(sqrt(2)/sqrt(3));
tau_pll_n1=2*xi_pll_n1/omega_pll_n1;
ki_pll1=kp_pll1/tau_pll_n1;

%% Transformer Impedance
R_trans=0.002;
L_trans=0.1/wb_n1;

%% Total PCC Impedance
R_pcc=R_trans+Rg_t;
L_pcc=L_trans+Xg_t/wb_n1;

%% Scenario configuration

Tsim=4;
T_stp=Tsim/2;

Vpeak_g1_init=1*(sqrt(2)/sqrt(3));  %change in grid nom voltage in p.u
Vpeak_g1_fnl=0.98*(sqrt(2)/sqrt(3));

angle_g1_init=0.0; %change in grid angle
angle_g1_fnl=0.0; 

