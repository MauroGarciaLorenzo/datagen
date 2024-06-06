%% run nonlinear and linear sim

param
sim('GFOL_nonlinear')
linear_model
% sim('GFOL_linear1')
sim('GFOL_linear2')

%% validation plots
x_min=Tsim/2-0.5;
x_max=2.8;

subpl_sz=[3 2];

valid_col = [0.7  0.7  0.7
             0.8500  0.3250  0.0980
             0.4660  0.6740  0.1880
             0.9290  0.6940  0.1250
             0  0.4470  0.7410];

label_fnt=14;
         
figure('Name','Comparison VSC: non linear, linear','position',[300,30,900,600]);


%Voltage u_q
subplot(subpl_sz(1),subpl_sz(2),1)
plot(tnolin,u_q_real_n1c/Vpeak_n1,'k','Color',valid_col(1,:),'linewidth',1.5)
hold all
plot(tlin,u_q_lineal2_n1c/Vpeak_n1,'-.k','Color',valid_col(2,:),'linewidth',1.5)
plot(time_pscad,u_q_PSCAD*1e3/Vpeak_n1,'-.k','Color',valid_col(3,:),'linewidth',1.5)
ylabel('$v^{q} \textrm{(pu)}$','Interpreter','Latex','Fontsize',label_fnt);
xlabel('time \textrm{(s)}','Interpreter','Latex','Fontsize',label_fnt);
xlim([x_min x_max])
% legend('Matlab non-linear', 'Matlab linear', 'PSCAD')
grid on;

%Voltage u_d
subplot(subpl_sz(1),subpl_sz(2),2)
plot(tnolin,u_d_real_n1c/Vpeak_n1,'k','Color',valid_col(1,:),'linewidth',1.5)
hold all
plot(tlin,u_d_lineal2_n1c/Vpeak_n1,'-.k','Color',valid_col(2,:),'linewidth',1.5)
plot(time_pscad,u_d_PSCAD*1e3/Vpeak_n1,'-.k','Color',valid_col(3,:),'linewidth',1.5)
ylabel('$v^{d} \textrm{(pu)}$','Interpreter','Latex','Fontsize',label_fnt);
xlabel('time \textrm{(s)}','Interpreter','Latex','Fontsize',label_fnt);
xlim([x_min x_max])
% legend('Matlab non-linear', 'Matlab linear', 'PSCAD')
grid on;

%current is_q
subplot(subpl_sz(1),subpl_sz(2),3)
plot(tnolin,is_q_real_n1c/Ipeak_n1,'k','Color',valid_col(1,:),'linewidth',1.5)
hold all
plot(tlin,is_q_lineal2_n1c/Ipeak_n1,'-.k','Color',valid_col(2,:),'linewidth',1.5)
plot(time_pscad,-is_q_PSCAD*1e3/Ipeak_n1,'-.k','Color',valid_col(3,:),'linewidth',1.5)
ylabel('$i_s^{q} \textrm{(pu)}$','Interpreter','Latex','Fontsize',label_fnt);
xlabel('time \textrm{(s)}','Interpreter','Latex','Fontsize',label_fnt);
xlim([x_min x_max])
% legend('Matlab non-linear', 'Matlab linear', 'PSCAD')
grid on;

%current is_d
subplot(subpl_sz(1),subpl_sz(2),4)
plot(tnolin,is_d_real_n1c/Ipeak_n1,'k','Color',valid_col(1,:),'linewidth',1.5)
hold all
plot(tlin,is_d_lineal2_n1c/Ipeak_n1,'-.k','Color',valid_col(2,:),'linewidth',1.5)
plot(time_pscad,-is_d_PSCAD*1e3/Ipeak_n1,'-.k','Color',valid_col(3,:),'linewidth',1.5)
ylabel('$i_s^{d} \textrm{(pu)}$','Interpreter','Latex','Fontsize',label_fnt);
xlabel('time \textrm{(s)}','Interpreter','Latex','Fontsize',label_fnt);
xlim([x_min x_max])
legend('Matlab non-linear', 'Matlab linear', 'PSCAD')
grid on;

%current fc
subplot(subpl_sz(1),subpl_sz(2),5)
plot(tnolin,freq_real_n1,'k','Color',valid_col(1,:),'linewidth',1.5)
hold all
plot(tlin,freq_lineal2_n1,'-.k','Color',valid_col(2,:),'linewidth',1.5)
plot(time_pscad,omega_VSC_PSCAD/(2*pi),'-.k','Color',valid_col(3,:),'linewidth',1.5)
ylabel('$f \textrm{(Hz)}$','Interpreter','Latex','Fontsize',label_fnt);
xlabel('time \textrm{(s)}','Interpreter','Latex','Fontsize',label_fnt);
xlim([x_min x_max])
% legend('Matlab non-linear', 'Matlab linear', 'PSCAD')
grid on;

