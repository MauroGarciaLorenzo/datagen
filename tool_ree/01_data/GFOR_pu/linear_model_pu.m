%% linearization points
% Base values
index0=199995;
% Ipeak_n1 = Sn_n1*sqrt(2)/(sqrt(3)*Un_n1);

% VSC 1

% 
is_qc0_n1=is_q_real_n1_pu(index0);
is_dc0_n1=is_d_real_n1_pu(index0);

u_qc0_n1=u_q_real_n1_pu(index0);
u_dc0_n1=u_d_real_n1_pu(index0);

vc_qc0_n1=vc_q_real_n1_pu(index0);
vc_dc0_n1=vc_d_real_n1_pu(index0);

ig_q0_n1=ig_q_real_n1_pu(index0);
ig_d0_n1=ig_d_real_n1_pu(index0);

ig_qc0_n1=igc_q_real_n1_pu(index0);
ig_dc0_n1=igc_d_real_n1_pu(index0);

f1_0=freq_real_n1_pu(index0);
w1_0=2*pi*f1_0;

% etheta0=3.82e-10;
etheta0=etheta_real_n1_pu(index0);


% AC grid 1

vg_q0_n1=vg_q_real_n1_pu(index0); 
vg_d0_n1=vg_d_real_n1_pu(index0);

vg_qc0_n1=vgc_q_real_n1_pu(index0);
vg_dc0_n1=vgc_d_real_n1_pu(index0);

% cap
ucap_d0_n1=u_dc0_n1+Rac_n1*(is_dc0_n1-ig_dc0_n1);
ucap_q0_n1=u_qc0_n1+Rac_n1*(is_qc0_n1-ig_qc0_n1);

%% State Space Models

% State-space VSC

% Frequency droop with low-pass filter in Pac:
Afdroop_n1=[-1/tau_droop_f];
Bfdroop_n1=[0 3*ig_q0_n1/2 3*ig_d0_n1/2 3*u_q0_n1/2 3*u_d0_n1/2];
Cfdroop_n1=[-k_droop_f_n1/tau_droop_f*wb_n1];
Dfdroop_n1=[+k_droop_f_n1*wb_n1 0 0 0 0];
fdroop_n1_x={'x_p1_filt'};
fdroop_n1_u={'P_ref_n1' 'u_q_n1' 'u_d_n1' 'ig_q_n1' 'ig_d_n1'};
fdroop_n1_y={'omega_n1'};
fdroop_n1 = ss(Afdroop_n1,Bfdroop_n1,Cfdroop_n1,Dfdroop_n1,'StateName',fdroop_n1_x,'inputname',fdroop_n1_u,'outputname',fdroop_n1_y);

% Voltage droop with low-pass filter in Qac:
Audroop_n1=[-1/tau_droop_u];
Budroop_n1=[0 -3*ig_d0_n1/2 3*ig_q0_n1/2 3*u_d0_n1/2 -3*u_q0_n1/2];
Cudroop_n1=[k_droop_u_n1/tau_droop_u];
Dudroop_n1=[+k_droop_u_n1 0 0 0 0];
udroop_n1_x={'x_q1_filt'};
udroop_n1_u={'Q_ref_n1' 'u_q_n1' 'u_d_n1' 'ig_q_n1' 'ig_d_n1'};
udroop_n1_y={'u_qc_ref_n1'};
udroop_n1 = ss(Audroop_n1,Budroop_n1,Cudroop_n1,Dudroop_n1,'StateName',udroop_n1_x,'inputname',udroop_n1_u,'outputname',udroop_n1_y);


% LC:
% Alc_n1=[-Rc_n1/Lc_n1 -w_n1 -1/Lc_n1 0;
%            w_n1 -Rc_n1/Lc_n1 0 -1/Lc_n1;
%            1/Cac_n1 0 0 -w_n1;
%            0 1/Cac_n1 w_n1 0];
% Blc_n1=[1/Lc_n1 0 0 0 -is_dc0_n1; %maybe ido??
%            0 1/Lc_n1 0 0 +is_qc0_n1;
%            0 0 -1/Cac_n1 0 -u_dc0_n1; 
%            0 0 0 -1/Cac_n1 +u_qc0_n1];
% Clc_n1=[1 0 0 0;
%            0 1 0 0;
%            0 0 1 0;
%            0 0 0 1];
% Dlc_n1=[0 0 0 0 0;
%            0 0 0 0 0;
%            0 0 0 0 0;
%            0 0 0 0 0];
% lc_n1_x={'is_q_n1' 'is_d_n1' 'u_q_n1' 'u_d_n1'};
% lc_n1_u={'vc_qc_n1' 'vc_dc_n1' 'ig_qc_n1' 'ig_dc_n1' 'omega_n1'};
% lc_n1_y={'is_qc_n1' 'is_dc_n1' 'u_qc_n1' 'u_dc_n1'};
% lc_n1 = ss(Alc_n1,Blc_n1,Clc_n1,Dlc_n1,'StateName',lc_n1_x,'inputname',lc_n1_u,'outputname',lc_n1_y);

% LC:
Alc_n1=[(-Rc_n1-Rac_n1)/Lc_n1 -wb_n1 -1/Lc_n1 0;
           wb_n1 (-Rc_n1-Rac_n1)/Lc_n1 0 -1/Lc_n1;
           1/Cac_n1 0 0 -wb_n1;
           0 1/Cac_n1 wb_n1 0];
Blc_n1=[1/Lc_n1 0 Rac_n1/Lc_n1 0 -is_d0_n1; %maybe ido??
           0 1/Lc_n1 0 Rac_n1/Lc_n1 +is_q0_n1;
           0 0 -1/Cac_n1 0 -ucap_d0_n1; 
           0 0 0 -1/Cac_n1 +ucap_q0_n1];
Clc_n1=[1 0 0 0;
           0 1 0 0;
           Rac_n1 0 1 0;
           0 Rac_n1 0 1];
Dlc_n1=[0 0 0 0 0;
           0 0 0 0 0;
           0 0 -Rac_n1 0 0;
           0 0 0 -Rac_n1 0];
lc_n1_x={'is_q_n1' 'is_d_n1' 'ucap_q_n1' 'ucap_d_n1'};
lc_n1_u={'vc_q_n1' 'vc_d_n1' 'ig_q_n1' 'ig_d_n1' 'omega_g1'};
lc_n1_y={'is_q_n1' 'is_d_n1' 'u_q_n1' 'u_d_n1'};
lc_n1 = ss(Alc_n1,Blc_n1,Clc_n1,Dlc_n1,'StateName',lc_n1_x,'inputname',lc_n1_u,'outputname',lc_n1_y);


% AC side voltage control:
Au_n1=[0 0;
         0 0];
Bu_n1=[1 0 -1 0 0 0;
         0 1 0 -1 0 0];
Cu_n1=[+ki_vac_n1 0;
         0 +ki_vac_n1];
Du_n1=[+kp_vac_n1 0 -kp_vac_n1 +wb_n1*Cac_n1 1 0;
         0 +kp_vac_n1 -wb_n1*Cac_n1 -kp_vac_n1 0 1];
u_n1_x={'Ke_u_q_n1','Ke_u_d_n1'};
u_n1_u={'u_qc_ref_n1','u_dc_ref_n1','u_qc_n1','u_dc_n1','ig_qc_f_n1','ig_dc_f_n1'};
u_n1_y={'is_qc_ref_n1','is_dc_ref_n1'};
u_n1 = ss(Au_n1,Bu_n1,Cu_n1,Du_n1,'StateName',u_n1_x,'inputname',u_n1_u,'outputname',u_n1_y);

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
is_n1_u={'is_qc_ref_n1' 'is_dc_ref_n1' 'is_qc_n1' 'is_dc_n1' 'u_qc_f_n1' 'u_dc_f_n1'};
is_n1_y={'vc_qc_n1' 'vc_dc_n1'};
is_n1 = ss(Ais_n1,Bis_n1,Cis_n1,Dis_n1,'StateName',is_n1_x,'inputname',is_n1_u,'outputname',is_n1_y);

% Conversion uc to ug: reverse
% Aug_n1=[0];
% Bug_n1=[0 0 0];
% Cug_n1=[0
%       0];
% Dug_n1=[cos(etheta0) sin(etheta0) -sin(etheta0)*u_qc0_n1+cos(etheta0)*u_dc0_n1;
%       -sin(etheta0) cos(etheta0) -cos(etheta0)*u_qc0_n1-sin(etheta0)*u_dc0_n1];
% ug_n1_x={''};
% ug_n1_u={'u_qc_n1','u_dc_n1' 'e_theta_n1'};
% ug_n1_y={'u_q_n1','u_d_n1'};
% ug_n1 = ss(Aug_n1,Bug_n1,Cug_n1,Dug_n1,'StateName',ug_n1_x,'inputname',ug_n1_u,'outputname',ug_n1_y);


%Conversion ig to igc:
Aigc_n1=[0];
Bigc_n1=[0 0 0];
Cigc_n1=[0
         0];
Digc_n1=[cos(etheta0) -sin(etheta0) -sin(etheta0)*ig_q0_n1-cos(etheta0)*ig_d0_n1;
         sin(etheta0) cos(etheta0) cos(etheta0)*ig_q0_n1-sin(etheta0)*ig_d0_n1];
igc_n1_x={''};
igc_n1_u={'ig_q_n1','ig_d_n1' 'e_theta_n1'};
igc_n1_y={'ig_qc_n1','ig_dc_n1'};
igc_n1 = ss(Aigc_n1,Bigc_n1,Cigc_n1,Digc_n1,'StateName',igc_n1_x,'inputname',igc_n1_u,'outputname',igc_n1_y);

% Conversion vg to vgc:
Avgc_n1=[0];
Bvgc_n1=[0 0 0];
Cvgc_n1=[0
         0];
Dvgc_n1=[cos(etheta0) -sin(etheta0) -sin(etheta0)*vg_q0_n1-cos(etheta0)*vg_d0_n1;
         sin(etheta0) cos(etheta0) cos(etheta0)*vg_q0_n1-sin(etheta0)*vg_d0_n1];
vgc_n1_x={''};
vgc_n1_u={'vg_q_n1','vg_d_n1' 'e_theta_n1'};
vgc_n1_y={'vg_qc_n1','vg_dc_n1'};
vgc_n1 = ss(Avgc_n1,Bvgc_n1,Cvgc_n1,Dvgc_n1,'StateName',vgc_n1_x,'inputname',vgc_n1_u,'outputname',vgc_n1_y);

% AC grid Thevenin model

Aig_n1=[-R_pcc/L_pcc -wb_n1;
           wb_n1 -R_pcc/L_pcc];
Big_n1=[-1/L_pcc 0 1/L_pcc 0 -ig_d0_n1;
           0 -1/L_pcc 0 1/L_pcc ig_q0_n1];
Cig_n1=[1 0;
        0 1];
Dig_n1=[0 0 0 0 0;
           0 0 0 0 0];
ig_n1_x={'ig_q_n1' 'ig_d_n1'};
ig_n1_u={'vg_qc_n1' 'vg_dc_n1' 'u_qc_n1' 'u_dc_n1' 'omega_n1'};
ig_n1_y={'ig_qc_n1' 'ig_dc_n1'};
ig_n1 = ss(Aig_n1,Big_n1,Cig_n1,Dig_n1,'StateName',ig_n1_x,'inputname',ig_n1_u,'outputname',ig_n1_y);

% omega to angle VSC (1/s)
Awc_n1=[0];
Bwc_n1=[1];
Cwc_n1=[1];
Dwc_n1=[0];
wc_n1_x={'x_omega_n1'};
wc_n1_u={'omega_n1'};
wc_n1_y={'angle_n1'};
wc_n1 = ss(Awc_n1,Bwc_n1,Cwc_n1,Dwc_n1,'StateName',wc_n1_x,'inputname',wc_n1_u,'outputname',wc_n1_y);

% omega to angle grid (1/s)
Awg_g1=[0];
Bwg_g1=[1 0];
Cwg_g1=[1];
Dwg_g1=[0 1];
wg_g1_x={'x_omega_g1'};
wg_g1_u={'omega_g1' 'grid_angle'};
wg_g1_y={'angle_g1'};
wg_g1 = ss(Awg_g1,Bwg_g1,Cwg_g1,Dwg_g1,'StateName',wg_g1_x,'inputname',wg_g1_u,'outputname',wg_g1_y);

% angle ref=angle_n1-angle_g1
Atheta_n1=[0];
Btheta_n1=[0 0];
Ctheta_n1=[0];
Dtheta_n1=[1 -1];
theta_n1_x={''};
theta_n1_u={'angle_n1' 'angle_g1'};
theta_n1_y={'e_theta_n1'};
theta_n1=ss(Atheta_n1,Btheta_n1,Ctheta_n1,Dtheta_n1,'StateName',theta_n1_x,'inputname',theta_n1_u,'outputname',theta_n1_y);

% AC voltage feedforward filter 
num_ig=1;
den_ig=[tau_ig 1];
[Af_ig,Bf_ig,Cf_ig,Df_ig]=tf2ss(num_ig,den_ig);
f_igd_x={'x_igd_ff'};
f_igd_u={'ig_dc_n1'};
f_igd_y={'ig_dc_f_n1'};
f_igd = ss(Af_ig,Bf_ig,Cf_ig,Df_ig,'StateName',f_igd_x,'inputname',f_igd_u,'outputname',f_igd_y);

f_igq_x={'x_igq_ff'};
f_igq_u={'ig_qc_n1'};
f_igq_y={'ig_qc_f_n1'};
f_igq = ss(Af_ig,Bf_ig,Cf_ig,Df_ig,'StateName',f_igq_x,'inputname',f_igq_u,'outputname',f_igq_y);

% current feedforward filter 
num_u=1;
den_u=1;
[Af_u,Bf_u,Cf_u,Df_u]=tf2ss(num_u,den_u);
f_ud_x={''};
f_ud_u={'u_dc_n1'};
f_ud_y={'u_dc_f_n1'};
f_ud = ss(Af_u,Bf_u,Cf_u,Df_u,'StateName',f_ud_x,'inputname',f_ud_u,'outputname',f_ud_y);

f_uq_x={''};
f_uq_u={'u_qc_n1'};
f_uq_y={'u_qc_f_n1'};
f_uq = ss(Af_u,Bf_u,Cf_u,Df_u,'StateName',f_uq_x,'inputname',f_uq_u,'outputname',f_uq_y);

%% Overall state-space model
ss_vsc = connect(lc_n1,fdroop_n1,udroop_n1,u_n1,is_n1,wc_n1,wg_g1,theta_n1,vgc_n1,...
                 ig_n1,f_igd,f_igq,f_ud,f_uq,...
                 {'P_ref_n1','Q_ref_n1','vg_q_n1','vg_d_n1','omega_g1','grid_angle'},...
                 {'u_qc_n1','u_dc_n1','vc_qc_n1','vc_dc_n1','is_qc_n1','is_dc_n1','ig_qc_n1','ig_dc_n1' });
ss_vsc=minreal(ss_vsc);             
% ss_vsc = connect(lc_n1,fdroop_n1,udroop_n1,u_n1,is_n1,wc_n1,wg_g1,theta_n1,vgc_n1,...
%                  ig_n1,f_igd,f_igq,f_ud,f_uq,...
%                  {'u_qc_ref_n1','u_dc_ref_n1'},...
%                  {'u_qc_n1','u_dc_n1' });
% ss_vsc=minreal(ss_vsc);              
