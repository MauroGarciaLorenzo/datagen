%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Grid-following 2L-VSC (PQ mode)
% Date: 25/05/2022
% CITCEA
% =========================================================================
% close all; clear all; clc
format long;
% =========================================================================

%% General parameters

% AC grid: Thevenin equivalent
Sn1=2.75e6;             % Base power
Un1=0.69e3;              % Base AC voltage
f1=50;                  % Base frequency
w_g1=2*pi*f1;


% only Z1 SCR=0.5
SCR1=0.5;                 % Short-circuit ratio
XR_ratio=3;
Scc1=SCR1*Sn1;          % Short-circuit power
Xcc1=Un1^2/Scc1;        % Short-circuit impedance
Lcc1=Xcc1/(2*pi*f1);    % Shot-circuit inductance
Rg_n1=sqrt(Xcc1^2/(XR_ratio^2+1));            % Thevenin resistance
Xg_n1=Rg_n1*XR_ratio;             
Lg_n1=Xg_n1/w_g1;             % Thevenin inductance

% Z1 and Z2 (parallel) equivalent SCR=3
SCR2=3;                
Scc2=SCR2*Sn1;          
Xcct=Un1^2/Scc2;        
Rg_t=sqrt(Xcct^2/(XR_ratio^2+1));          
Xg_t=Rg_t*XR_ratio;             
Lg_t=Xg_t/w_g1;

Zg_n2=1/(1/complex(Rg_t,Xg_t)-1/complex(Rg_n1,Xg_n1));
Rg_n2=real(Zg_n2);            % Thevenin resistance
Lg_n2=imag(Zg_n2)/w_g1;             % Thevenin inductance


% DC grid: ideal
Vdcref_n1=2e3;   

% Other parameters
sat_current=1.15;        % Current saturation
epsilon=1;              % Used for the current saturation
v_ini=1e-5;             % Initialization voltages
magnitud_delay=1e-5;    % Constant delay for simulation    
plots_step=10e-6;        % Plot samples step 

%% 2L-VSC 1 (PQ mode)

Sn_n1=Sn1;                           % Base power
cosfi_n1=1;
f_n1=f1;
w_n1=2*pi*f_n1;
Un_n1=Un1;                          % Line-line voltage
Vpeak_n1=Un_n1/sqrt(3)*sqrt(2);     % Peak voltage
Ipeak_n1=Sn_n1/(sqrt(3)*Un_n1);
Xn_n1=Un_n1^2/Sn_n1;                % Base impedance
Ln_n1=Xn_n1/(2*pi*f_n1);            % Base inductance


Rc_n1=0.005*Xn_n1;                   % Converter grid coupling filter resistance
Lc_n1=0.15*Ln_n1;                    % Converter grid coupling filter inductance
Cac_n1=0.075/Xn_n1/w_n1;     % Converter grid coupling filter capacitance
Rac_n1=1*1/(3*10*w_n1*Cac_n1);     % passive damping 

% vaini_n1=Vpeak_n1;                  % PCC initial voltages
% vbini_n1=-1/2*Vpeak_n1;
% vcini_n1=-1/2*Vpeak_n1;

P_ref_n1=0.5*Sn_n1;
Q_ref_n1=0.1*Sn_n1;


% PLL tuning
ts_pll_n1=0.1;
xi_pll_n1=0.707;
omega_pll_n1=4/(ts_pll_n1*xi_pll_n1);
kp_pll_n1=xi_pll_n1*2*omega_pll_n1/Vpeak_n1;
tau_pll_n1=2*xi_pll_n1/omega_pll_n1;
ki_pll_n1=kp_pll_n1/tau_pll_n1;

% Current control
taus=1e-3;
kp_s_n1=Lc_n1/taus;
ki_s_n1=Rc_n1/taus;

% Power loops
tau_p=0.1;
kp_P_n1=1e-3*taus/tau_p;
ki_P_n1=1e-3*1/tau_p;
tau_q=0.1;
kp_Q_n1=1e-3*taus/tau_q;
ki_Q_n1=1e-3*1/tau_q;

% P-w droop
tau_droop_f=1/50;
k_droop_f_n1=20*Sn_n1/w_n1;

% V-Q droop
tau_droop_u=1/50;
k_droop_u_n1=0.3*50*Sn_n1/Vpeak_n1;

%% Scenario configuration and linearisation:

Tsim=4;
T_step=2;
T_solver=10e-6;

Vpeak_q1_init=Vpeak_n1*1;
Vpeak_g1_fnl=Vpeak_n1*0.98;
% 
% tau_ramp=25e-3;
% t_ramp=0.05;
% t_step_P=0.50;
% t_step_Q=0.75;
% 
% P_set=0.9*Sn_n1;
% Q_set=0.2*Sn_n1;
% P_step=P_set*1.01; % Linear model validation purposes
% Q_step=Q_set*1.01;
% 
% sim('TwoLevelVSC_PQ')
% 
% index0=find(tnolin>=t_step_P-0.001,1);
% linearization_point
% linear_model

%% Linear model validation
% 
% sim ('TwoLevelVSC_PQ_linear1') % Linear model in blocks form
% sim ('TwoLevelVSC_PQ_linear2') % Linear model in state-space form
%      
% linear_validation
