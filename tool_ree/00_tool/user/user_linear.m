%% IEEE 9-bus system

tstep = 150-data_PSCAD.time(1);%70; %180; 
Tsim  = 170-data_PSCAD.time(1);%90; %200; 

tstep_lin = 0.05; %0.05;
Tsim_lin = Tsim-tstep;


% Define case study for linear model simulation

%% Case #1 : Disturbance in Thevenin voltage reference

% deltax = 0.01*1.04;

%% Case #1b: Test small model
% 
Rload_init = T_load.R(T_load.bus == bus_in);
DR = 0.01;
DRload = -DR/(1+DR)*Rload_init;
deltax = DRload;


% deltax = u_qc0*0.01;
% deltax =(188.9566-187.0858)*1000/Vb_vsc
% deltax = (-0.017874--0.018052740000000)*1000/Ib_vsc; %step ic_q_ref


% IEEE-118 bus
% deltax = results.bus{results.bus.bus==bus_in,"Vm"}*0.01;