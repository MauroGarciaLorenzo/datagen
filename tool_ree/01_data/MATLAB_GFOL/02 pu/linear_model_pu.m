%% Operation Points

index0=T_step*99990;

is_q0_n1g=is_q_real_n1g_pu(index0);
is_d0_n1g=is_d_real_n1g_pu(index0);

u_q0_n1g=u_q_real_n1g_pu(index0);
u_d0_n1g=u_d_real_n1g_pu(index0);

% vc_q0_n1g=vc_q_real_n1g(index0);
% vc_d0_n1g=vc_d_real_n1g(index0);

is_q0_n1c=is_q_real_n1c_pu(index0);
is_d0_n1c=is_d_real_n1c_pu(index0);

u_q0_n1c=u_q_real_n1c_pu(index0);
u_d0_n1c=u_d_real_n1c_pu(index0);

vc_q0_n1c=vc_q_real_n1c_pu(index0);
vc_d0_n1c=vc_d_real_n1c_pu(index0);

freq0_real_n1=freq_real_n1_pu(index0);
etheta0_n1=etheta_n1_pu(index0)*1;

%% State-space

% PLL:
Apll_n1=[0];
Bpll_n1=[1];
Cpll_n1=[-ki_pll_n1*wb_n1];
Dpll_n1=[-kp_pll_n1*wb_n1];
pll_n1_x={'x_pll_n1'};
pll_n1_u={'u_dc_n1'};
pll_n1_y={'omega_pll_n1'};
pll_n1 = ss(Apll_n1,Bpll_n1,Cpll_n1,Dpll_n1,'StateName',pll_n1_x,'inputname',pll_n1_u,'outputname',pll_n1_y);

% e_theta (PLL-Grid):
Atheta_n1=[0];
Btheta_n1=[1 -1];
Ctheta_n1=[1];
Dtheta_n1=[0 0];
theta_n1_x={'e_theta_n1'};
theta_n1_u={'omega_pll_n1' 'omega_n1'};
theta_n1_y={'e_theta_n1'};
theta_n1 = ss(Atheta_n1,Btheta_n1,Ctheta_n1,Dtheta_n1,'StateName',theta_n1_x,'inputname',theta_n1_u,'outputname',theta_n1_y);

% Conversion is to isc
Aisc_n1=[0];
Bisc_n1=[0 0 0];
Cisc_n1=[0
         0];
Disc_n1=[cos(etheta0_n1) -sin(etheta0_n1) -sin(etheta0_n1)*is_q0_n1g-cos(etheta0_n1)*is_d0_n1g;
         sin(etheta0_n1) cos(etheta0_n1) cos(etheta0_n1)*is_q0_n1g-sin(etheta0_n1)*is_d0_n1g];
isc_n1_x={''};
isc_n1_u={'is_q_n1' 'is_d_n1' 'e_theta_n1'};
isc_n1_y={'is_qc_n1' 'is_dc_n1'};
isc_n1 = ss(Aisc_n1,Bisc_n1,Cisc_n1,Disc_n1,'StateName',isc_n1_x,'inputname',isc_n1_u,'outputname',isc_n1_y);

% Conversion u to uc
Auc_n1=[0];
Buc_n1=[0 0 0];
Cuc_n1=[0
        0];
Duc_n1=[cos(etheta0_n1) -sin(etheta0_n1) -sin(etheta0_n1)*u_q0_n1g-cos(etheta0_n1)*u_d0_n1g;
         sin(etheta0_n1) cos(etheta0_n1) cos(etheta0_n1)*u_q0_n1g-sin(etheta0_n1)*u_d0_n1g];
uc_n1_x={''};
uc_n1_u={'u_q_n1' 'u_d_n1' 'e_theta_n1'};
uc_n1_y={'u_qc_n1' 'u_dc_n1'};
uc_n1 = ss(Auc_n1,Buc_n1,Cuc_n1,Duc_n1,'StateName',uc_n1_x,'inputname',uc_n1_u,'outputname',uc_n1_y);

% Conversion vcc to vc
Avcc_n1=[0];
Bvcc_n1=[0 0 0];
Cvcc_n1=[0
            0];
Dvcc_n1=[cos(etheta0_n1) sin(etheta0_n1) -sin(etheta0_n1)*vc_q0_n1c+cos(etheta0_n1)*vc_d0_n1c;
            -sin(etheta0_n1) cos(etheta0_n1) -cos(etheta0_n1)*vc_q0_n1c-sin(etheta0_n1)*vc_d0_n1c];
vcc_n1_x={''};
vcc_n1_u={'vc_qc_n1' 'vc_dc_n1' 'e_theta_n1'};
vcc_n1_y={'vc_q_n1' 'vc_d_n1'};
vcc_n1 = ss(Avcc_n1,Bvcc_n1,Cvcc_n1,Dvcc_n1,'StateName',vcc_n1_x,'inputname',vcc_n1_u,'outputname',vcc_n1_y);

% LCL
% Alcl_n1=[-Rc_n1/Lc_n1 -wb_n1 -1/Lc_n1 0 0 0;
%         +wb_n1 -Rc_n1/Lc_n1 0 -1/Lc_n1 0 0;
%         1/Cac_n1 0 0 -wb_n1 -1/Cac_n1 0;
%         0 1/Cac_n1 +wb_n1 0 0 -1/Cac_n1;
%         0 0 1/Lg_n1 0 -Rg_n1/Lg_n1 -wb_n1;
%         0 0 0 1/Lg_n1 +wb_n1 -Rg_n1/Lg_n1];
% Blcl_n1=[1/Lc_n1 0 0 0;
%         0 1/Lc_n1 0 0;
%         0 0 0 0;
%         0 0 0 0;
%         0 0 -1/Lg_n1 0 ;
%         0 0 0 -1/Lg_n1];
% Clcl_n1=[1 0 0 0 0 0;
%         0 1 0 0 0 0;
%         0 0 1 0 0 0;
%         0 0 0 1 0 0;
%         0 0 0 0 1 0;
%         0 0 0 0 0 1];
% Dlcl_n1=[0 0 0 0;
%         0 0 0 0;
%         0 0 0 0;
%         0 0 0 0;
%         0 0 0 0;
%         0 0 0 0];
% lcl_n1_x={'is_q_n1' 'is_d_n1' 'u_q_n1' 'u_d_n1' 'ig_q_n1' 'ig_d_n1'};
% lcl_n1_u={'vc_q_n1' 'vc_d_n1' 'vg_q_n1' 'vg_d_n1'};
% lcl_n1_y={'is_q_n1' 'is_d_n1' 'u_q_n1' 'u_d_n1' 'ig_q_n1' 'ig_d_n1'};
% lcl_n1=ss(Alcl_n1,Blcl_n1,Clcl_n1,Dlcl_n1,'StateName',lcl_n1_x,'inputname',lcl_n1_u,'outputname',lcl_n1_y);

% plant: RL filter and AC grid thevenin
R_pscad=5e-4; % equivalent PSCAD resistance used in nonlinear model
Arl_n1=[-(Rc_n1+Rg_t+R_pscad)/(Lc_n1+Lg_t) -wb_n1;
             wb_n1 -(Rc_n1+Rg_t+R_pscad)/(Lc_n1+Lg_t)];
Brl_n1=[+1/(Lc_n1+Lg_t) 0 -1/(Lc_n1+Lg_t) 0;
             0 +1/(Lc_n1+Lg_t) 0 -1/(Lc_n1+Lg_t)];
Crl_n1=[1 0;
             0 1;
             (Lc_n1*Rg_t-Lg_t*(Rc_n1+R_pscad))/(Lc_n1+Lg_t) 0;
             0 (Lc_n1*Rg_t-Lg_t*(Rc_n1+R_pscad))/(Lc_n1+Lg_t)];
Drl_n1=[0 0 0 0;
             0 0 0 0;
         	 Lg_t/(Lc_n1+Lg_t) 0 Lc_n1/(Lc_n1+Lg_t) 0;
             0 Lg_t/(Lc_n1+Lg_t) 0 Lc_n1/(Lc_n1+Lg_t)];
rl_n1_x={'is_q_n1' 'is_d_n1'};
rl_n1_u={'vc_q_n1' 'vc_d_n1' 'vg_q_n1' 'vg_d_n1'};
rl_n1_y={'is_q_n1' 'is_d_n1' 'u_q_n1' 'u_d_n1'};
rl_n1 = ss(Arl_n1,Brl_n1,Crl_n1,Drl_n1,'StateName',rl_n1_x,'inputname',rl_n1_u,'outputname',rl_n1_y);

% AC side current control
Ais_n1=[0 0;
        0 0];
Bis_n1=[1 0 -1 0 0 0;
        0 1 0 -1 0 0];
Cis_n1=[+ki_s_n1 0;
        0 +ki_s_n1];
Dis_n1=[+kp_s_n1 0 -kp_s_n1 +wb_n1*Lc_n1 1 0;
        0 +kp_s_n1 -wb_n1*Lc_n1 -kp_s_n1 0 1];
is_n1_x={'Ke_is_q_n1' 'Ke_is_d_n1'};
is_n1_u={'is_q_ref_n1' 'is_d_ref_n1' 'is_qc_n1' 'is_dc_n1' 'u_qc_n1' 'u_dc_n1'};
is_n1_y={'vc_qc_n1' 'vc_dc_n1'};
is_n1 = ss(Ais_n1,Bis_n1,Cis_n1,Dis_n1,'StateName',is_n1_x,'inputname',is_n1_u,'outputname',is_n1_y);

% Frequency droop with low-pass filter on omega:
Afdroop_n1=[-1/tau_droop_f];
Bfdroop_n1=[0 1];
Cfdroop_n1=[-k_droop_f_n1/tau_droop_f/wb_n1];
Dfdroop_n1=[+k_droop_f_n1/wb_n1 0];
fdroop_n1_x={'x_f_filt'};
fdroop_n1_u={'omega_ref_n1' 'omega_pll_n1'};
fdroop_n1_y={'P_ref_n1'};
fdroop_n1 = ss(Afdroop_n1,Bfdroop_n1,Cfdroop_n1,Dfdroop_n1,'StateName',fdroop_n1_x,'inputname',fdroop_n1_u,'outputname',fdroop_n1_y);

% voltage magnitude
Au_n1=[0];
Bu_n1=[0 0];
Cu_n1=[0];
Du_n1=[u_q0_n1c/(sqrt(u_q0_n1c^2+u_d0_n1c^2)) u_d0_n1c/(sqrt(u_q0_n1c^2+u_d0_n1c^2))];
Au_n1_x={''};
Au_n1_u={'u_qc_n1' 'u_dc_n1'};
Au_n1_y={'u_n1'};
u_n1 = ss(Au_n1,Bu_n1,Cu_n1,Du_n1,'StateName',Au_n1_x,'inputname',Au_n1_u,'outputname',Au_n1_y);

% Voltage droop with low-pass filter in v:
Audroop_n1=[-1/tau_droop_u];
Budroop_n1=[0 1];
Cudroop_n1=[-k_droop_u_n1/tau_droop_u];
Dudroop_n1=[+k_droop_u_n1 0];
udroop_n1_x={'x_q1_filt'};
udroop_n1_u={'u_ref_n1' 'u_n1'};
udroop_n1_y={'Q_ref_n1'};
udroop_n1 = ss(Audroop_n1,Budroop_n1,Cudroop_n1,Dudroop_n1,'StateName',udroop_n1_x,'inputname',udroop_n1_u,'outputname',udroop_n1_y);


% P control
Ap_n1=[0];
Bp_n1=[1 -3/2*u_q0_n1c -3/2*u_d0_n1c -3/2*is_q0_n1c -3/2*is_d0_n1c];
Cp_n1=[ki_P_n1];
Dp_n1=[kp_P_n1 -3/2*u_q0_n1c*kp_P_n1 -3/2*u_d0_n1c*kp_P_n1 -3/2*is_q0_n1c*kp_P_n1 -3/2*is_d0_n1c*kp_P_n1];
p_n1_x={'Ke_P_n1'};
p_n1_u={'P_ref_n1' 'is_qc_n1' 'is_dc_n1' 'u_qc_n1' 'u_dc_n1'};
p_n1_y={'is_q_ref_n1'};
p_n1 = ss(Ap_n1,Bp_n1,Cp_n1,Dp_n1,'StateName',p_n1_x,'inputname',p_n1_u,'outputname',p_n1_y);

% Q control
Aq_n1=[0];
Bq_n1=[1 +3/2*u_d0_n1c -3/2*u_q0_n1c -3/2*is_d0_n1c +3/2*is_q0_n1c];
Cq_n1=[ki_Q_n1];
Dq_n1=[kp_Q_n1 +3/2*u_d0_n1c*kp_Q_n1 -3/2*u_q0_n1c*kp_Q_n1 -3/2*is_d0_n1c*kp_Q_n1 +3/2*is_q0_n1c*kp_Q_n1];
q_n1_x={'Ke_Q_n1'};
q_n1_u={'Q_ref_n1' 'is_qc_n1' 'is_dc_n1' 'u_qc_n1' 'u_dc_n1'};
q_n1_y={'is_d_ref_n1'};
q_n1 = ss(Aq_n1,Bq_n1,Cq_n1,Dq_n1,'StateName',q_n1_x,'inputname',q_n1_u,'outputname',q_n1_y);


%% Overal Linear Model
ss_sys = connect(pll_n1,theta_n1,isc_n1,uc_n1,vcc_n1,rl_n1,is_n1,p_n1,q_n1,...
                 fdroop_n1,u_n1,udroop_n1,...
                 {'omega_ref_n1' 'u_ref_n1' 'vg_q_n1' 'vg_d_n1'},...
                 {'is_q_n1' 'is_d_n1'});
ss_sys=minreal(ss_sys); 

