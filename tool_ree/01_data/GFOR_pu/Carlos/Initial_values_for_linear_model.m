%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save initial values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Sb = results.baseMVA;
fref = 50;

n_sg = data.SG(:,1);
ii_gfol = find(cases(i_case,:)==0);
n_gfol = data.VSC(ii_gfol,1);
ii_gfor = find(cases(i_case,:)==1);
n_gfor = data.VSC(ii_gfor,1);
Ssg_base = results.gen(n_sg,9);
In_sg = Ssg_base*1e6/Vhv*sqrt(2/3);

%===============================================================================================
% SYNCHRONOUS GENERATOR
%===============================================================================================
%Results from power flow
Psg0 = results.gen(n_sg,2)./Ssg_base; 
Qsg0 = results.gen(n_sg,3)./Ssg_base;
Vsg = results.bus(n_sg,8);
delta_bus = 0;%results.bus(n_sg,9)/180*pi;

%SG current
I = abs(conj((Psg0+i*Qsg0)./Vsg));
phi = -acos(Psg0./(sqrt(Psg0.^2+Qsg0.^2))).*sign(Qsg0./Psg0);

% Internal voltage
E = (Vsg.*cos(delta_bus)+i*Vsg.*sin(delta_bus))+(Rs_pu+i*Xq)*(I.*cos(phi)+i*I.*sin(phi));
Eq = real(E);
Ed = imag(E);

Emag = abs(E);
delta = atan(Ed./Eq); %rotor angle

% qd currents
Iq = I.*cos(-delta+phi);
Id = -I.*sin(-delta+phi);

% qd terminal voltage
Vq = Vsg.*cos(delta_bus-delta);
Vd = -Vsg.*sin(delta_bus-delta);

% Field voltage
Eq_tr = Emag-Id*(Xq-Xd_tr);
Efd = Eq_tr + (Xd-Xd_tr)*Id;
Ifd = Efd/Lmd_pu;

% Governor
Pm0 = Psg0+(Iq.^2+Id.^2)*Rs_pu;

% Initial values for linear model
isq0 = Iq;  % pu
isd0 = Id;  % pu
ifd0 = Ifd; % pu
ikd0 = zeros(length(n_sg),1);
ikq10 = zeros(length(n_sg),1);
ikq20 = zeros(length(n_sg),1);
vsgq_pu0 = Vq; % V
vsgd_pu0 = Vd; % V
w0_pu = 1;%results.f/fref; % pu
w0 = wn;%results.f*2*pi; % rad/s
% e_theta0 = results.bus(1:2,9)*pi/180+delta; % if voltage source as slack
e_theta0 = results.bus(1:2,9)*pi/180+delta-delta(1); % if SG1 as slack


%% ===============================================================================================
% GRID
%===============================================================================================
% Bus voltages
% th = delta*180/pi-90+results.bus(ng,9);
% theta0 = results.bus(:,9)*pi/180; % if voltage source as slack
theta0 = results.bus(:,9)*pi/180-delta(1); % if SG1 as slack
vn = results.bus(:,8)*Vhv;

% As vectors
vb_q0 = vn.*cos(theta0)*sqrt(2/3);
vb_d0 = -vn.*sin(theta0)*sqrt(2/3);

% Branch currents (transfomers and lines)
ibranch_q0 = Sb*1e6*results_branch(:,3)./vn(results_branch(:,1))*sqrt(2/3);
ibranch_d0 = Sb*1e6*results_branch(:,4)./vn(results_branch(:,1))*sqrt(2/3);
[ibranch_q0,ibranch_d0] = rotation_vect(ibranch_q0,ibranch_d0,-(theta0(results_branch(:,1))));

%Load (inductante) currents
n_ld = data.load(:,1);
Xload = data.load(:,3)*wn;
idL = -(vb_q0(n_ld) + i*vb_d0(n_ld))./(i*Xload);
idL_q0 = real(idL);
idL_d0 = imag(idL);

% SG current in local reference
[isqg0,isdg0] = rotation_vect(isq0,isd0,delta);

%%
%===============================================================================================
% GFOL
%===============================================================================================

Svsc_base = results.gen(n_gfol,9);
%Results from power flow
Pvsc0 = results.gen(n_gfol,2)*1e6; 
Qvsc0 = results.gen(n_gfol,3)*1e6;
Vvsc = results.bus(n_gfol,8)*Vhv;
% e_theta_vsc0 = results.bus(n_gfol,9)/180*pi; % if voltage source as slack
e_theta_vsc0 = results.bus(n_gfol,9)/180*pi-delta(1); % if SG1 as slack

Ivsc0 = sqrt(Pvsc0.^2+Qvsc0.^2)./Vvsc*sqrt(2/3);

phi_c = acos(Pvsc0./sqrt(Pvsc0.^2+Qvsc0.^2)).*sign(Qvsc0./Pvsc0);

vpccqc0 = Vvsc*sqrt(2/3);
vpccdc0 = zeros(length(n_gfol),1);

[vpccq0,vpccd0] = rotation_vect(vpccqc0, vpccdc0,-e_theta_vsc0);

icqc0 = Ivsc0.*cos(phi_c);
icdc0 = Ivsc0.*sin(phi_c);

[icq0,icd0] = rotation_vect(icqc0, icdc0,-e_theta_vsc0);

for ii = 1:length(ii_gfol)
    vconvqc0(ii) = vpccqc0(ii)-VSC(ii_gfol(ii)).Rc*icqc0(ii)+VSC(ii_gfol(ii)).Lc*wn*icdc0(ii);
    vconvdc0(ii) = vpccdc0(ii)-VSC(ii_gfol(ii)).Rc*icdc0(ii)-VSC(ii_gfol(ii)).Lc*wn*icqc0(ii);
end

%===============================================================================================
% GFOR
%===============================================================================================

%Results from power flow
Pgfor0 = results.gen(n_gfor,2)*1e6; 
Qgfor0 = results.gen(n_gfor,3)*1e6;
Vgfor0 = results.bus(n_gfor,8)*Vhv;
% e_theta_vsc0 = results.bus(n_gfor,9)/180*pi; % if voltage source as slack
e_theta_gfor0 = results.bus(n_gfor,9)/180*pi-delta(1); % if SG1 as slack

Igfor0 = sqrt(Pgfor0.^2+Qgfor0.^2)./Vgfor0*sqrt(2/3);

phi_gfor = acos(Pgfor0./sqrt(Pgfor0.^2+Qgfor0.^2)).*sign(Qgfor0./Pgfor0);

ilqc_gfor0 = Igfor0.*cos(phi_gfor);
ildc_gfor0 = Igfor0.*sin(phi_gfor);

[ilq_gfor0,ild_gfor0] = rotation_vect(ilqc_gfor0, ildc_gfor0,-e_theta_gfor0);

vpccqc_gfor0 = Vgfor0*sqrt(2/3);
vpccdc_gfor0 = zeros(length(ii_gfor),1);

[vpccq_gfor0,vpccd_gfor0] = rotation_vect(vpccqc_gfor0, vpccdc_gfor0,-e_theta_gfor0);

for ii = 1:length(ii_gfor)
    icq_gfor0(ii) = ilq_gfor0(ii) + VSC(ii_gfor(ii)).Cc*w0*vpccd_gfor0(ii);
    icd_gfor0(ii) = ild_gfor0(ii) - VSC(ii_gfor(ii)).Cc*w0*vpccq_gfor0(ii);
    [icqc_gfor0(ii),icdc_gfor0(ii)] = rotation_vect(icq_gfor0(ii), icd_gfor0(ii),e_theta_gfor0(ii));
    vconvqc_gfor0(ii) = vpccqc_gfor0(ii)+(VSC(ii_gfor(ii)).Rc*icqc_gfor0(ii)+VSC(ii_gfor(ii)).Lc*wn*icdc_gfor0(ii));
    vconvdc_gfor0(ii) = vpccdc_gfor0(ii)+(VSC(ii_gfor(ii)).Rc*icdc_gfor0(ii)-VSC(ii_gfor(ii)).Lc*wn*icqc_gfor0(ii));
end