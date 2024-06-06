%% COMPARE IZ-RL and PI-BERGERON behaviour in IEEE-9bus system 

time = 0:1e-4:Tsim_lin;
nSamples = length(time);

% Tinf = tstep-0.05;
% Tsup = tstep+0.2;

%% DEFINE plot parameters

c1 = [0.4510    0.7922    0.9412];
c2 = [0.1725    0.6078    0.9020];
c3 = [0    0.3647    0.6118];
colors_pi = {c1 c3};

c1 = [0.8000    0.8000    0.8000];
c2 = [0.6510    0.6510    0.6510];
c3 = [0.5020    0.5020    0.5020];
colors_berg = {c1 c3};
colors = [colors_pi(:)' colors_berg(:)'];

gray = {[0.8 0.8 0.8],[0.5020 0.5020 0.5020]};
color_lin   = {[0.2784    0.6275    0.9294],[0    0.4471    0.7412]};
color_nolin = {[0.9294    0.4745    0.2784],[0.8510 0.3255 0.0980]};

pos211 = [0.25 0.6 0.65 0.38];
pos212 = [0.25 0.15  0.65 0.38];

pos221 = [0.15 0.60  0.3 0.38];
pos222 = [0.60 0.60  0.3 0.38];
pos223 = [0.15 0.15  0.3 0.38];
pos224 = [0.60 0.15  0.3 0.38];

set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

   
%% Read data

    filePSCAD = 'data_9bus_rl_pi.txt';
    data_PSCAD_rl_pi = readtable([path_pscad filePSCAD]);
    filePSCAD = 'data_9bus_iz_pi.txt';
    data_PSCAD_iz_pi = readtable([path_pscad  filePSCAD]);
    filePSCAD = 'data_9bus_rl_berg.txt';
    data_PSCAD_rl_berg = readtable([path_pscad  filePSCAD]);
    filePSCAD = 'data_9bus_iz_berg.txt';
    data_PSCAD_iz_berg = readtable([path_pscad  filePSCAD]);

%     filePSCAD = 'data_9bus_rl_berg_bus4.txt';
%     data_PSCAD_rl_berg4 = readtable([path_pscad  filePSCAD]);

    data_PSCAD_array = {data_PSCAD_rl_berg, data_PSCAD_iz_berg, data_PSCAD_rl_pi, data_PSCAD_iz_pi};

%     ss_sys_rl = ss_sys;
%     out_lin_rl = out_lin; 
%     save([path_results 'out_lin_rl.mat'], 'out_lin_rl');
%     load([path_results 'out_lin_rl.mat'])

%     ss_sys_iz = ss_sys;
%     out_lin_iz = out_lin;
%     save([path_results 'out_lin_iz.mat'], 'out_lin_iz');
%     load([path_results 'out_lin_iz.mat'])

%% Compare RL load with BERG-RL / BERG-IZ
    out_lin = out_lin_iz;
    fuq = figure;
    fud = figure;

    % plot linear iz
        data_PSCAD = data_PSCAD_iz_berg;
        % Bus 2: voltages / PSCAD: linearization point from PSCAD
        vq0 = data_PSCAD.L_L_Uq(find(data_PSCAD.time>=tstep-0.01,1));
        vd0 = data_PSCAD.L_L_Ud(find(data_PSCAD.time>=tstep-0.01,1));
    
        vn2q_lin = out_lin.NET_vn2q + vq0*ones(nSamples,1);
        vn2d_lin = out_lin.NET_vn2d + vd0*ones(nSamples,1);
        
        figure(fuq) 
            hold on
            plot(time,vn2q_lin,'LineWidth',1.3,'Color',gray)
        figure(fud)
            hold on
            plot(time,vn2d_lin,'LineWidth',1.3,'Color',gray)


    out_lin = out_lin_rl;
    for idx = 1:2
        if idx == 1
            data_PSCAD = data_PSCAD_rl_berg;
        elseif idx == 2
            data_PSCAD = data_PSCAD_iz_berg;
        end

        % Bus 2: voltages / PSCAD: linearization point from PSCAD
        vq0 = data_PSCAD.L_L_Uq(find(data_PSCAD.time>=tstep-0.01,1));
        vd0 = data_PSCAD.L_L_Ud(find(data_PSCAD.time>=tstep-0.01,1));
    
        vn2q_lin = out_lin.NET_vn2q + vq0*ones(nSamples,1);
        vn2d_lin = out_lin.NET_vn2d + vd0*ones(nSamples,1);
        
        figure(fuq) 
            hold on
            plot(time,vn2q_lin,'LineWidth',1.3,'Color',color_lin{idx})
            hold on
            plot(data_PSCAD.time,data_PSCAD.L_L_Uq,'LineWidth',1.3,'Color',color_nolin{idx})
    
        figure(fud)
            hold on
            plot(time,vn2d_lin,'LineWidth',1.3,'Color',color_lin{idx})
            hold on
            plot(data_PSCAD.time,data_PSCAD.L_L_Ud,'LineWidth',1.3,'Color',color_nolin{idx})
    end

        figure(fuq)
            hold on
            xlim([tstep-0.05 tstep+1])
            legend('linear-IZ','linear-RL','PSCAD-RL-BERG','linear-RL','PSCAD-IZ-BERG')
            title('PSCAD-RL-BERG vs PSCAD-IZ-BERG')
            grid on
            ylabel('$U_{q}$','Interpreter','latex','FontName','Times New Roman','FontAngle','normal','FontWeight','bold','FontSize',10)
            set(gcf,'Position',[100 100 300 300])
    
        figure(fud)
            hold on
            xlim([tstep-0.05 tstep+1])        
            legend('linear-IZ','linear-RL','PSCAD-RL-BERG','linear-RL','PSCAD-IZ-BERG')
            title('PSCAD-RL-BERG vs PSCAD-IZ-BERG')
            grid on
            ylabel('$U_{d}$','Interpreter','latex','FontName','Times New Roman','FontAngle','normal','FontWeight','bold','FontSize',10)
            set(gcf,'Position',[100 100 300 300])

%             exportgraphics(gcf,join(['02_nonlinear_linear/' caseName '/MMC1_vpq' '.pdf']))
%             exportgraphics(gcf,join(['02_nonlinear_linear/' caseName '/MMC1_vpq' '.png']))

%% compare one by one     
    
    data_PSCAD = data_PSCAD_iz_pi;
    out_lin = out_lin_iz;
    
    % Bus 1 (4 in 9bus): voltages / linearization point from PSCAD
    vq0 = data_PSCAD.Uq(find(data_PSCAD.time>=tstep-0.01,1));
    vd0 = data_PSCAD.Ud(find(data_PSCAD.time>=tstep-0.01,1));

    vnq_lin = out_lin.NET_vn1q + vq0*ones(nSamples,1);
    vnd_lin = out_lin.NET_vn1d + vd0*ones(nSamples,1);


    fuq = figure;
    fud = figure;
        figure(fuq) 
            plot(time,vnq_lin,'LineWidth',1.3,'Color',color_lin{1})
            hold on
            plot(data_PSCAD.time,data_PSCAD.Uq,'LineWidth',1.3,'Color',color_nolin{1})
        figure(fud)
            plot(time,vnd_lin,'LineWidth',1.3,'Color',color_lin{1})
            hold on
            plot(data_PSCAD.time,data_PSCAD.Ud,'LineWidth',1.3,'Color',color_nolin{1})
            
   
        figure(fuq)
            hold on
            xlim([Tinf Tsup])
            legend('linear-IZ','PSCAD-PI')
            set(gcf,'color','w');
            %title('best match case')
            grid on
            title('$v_{4q}$ [pu]','Interpreter','latex','FontName','Times New Roman','FontAngle','normal','FontWeight','bold','FontSize',12)
            set(gcf,'Position',[100 100 300 300])
    
        figure(fud)
            hold on
            xlim([Tinf Tsup])      
            legend('linear-IZ','PSCAD-IZ-PI')
            set(gcf,'color','w');
            %title('best match case')
            grid on
            title('$v_{4d}$ [pu]','Interpreter','latex','FontName','Times New Roman','FontAngle','normal','FontWeight','bold','FontSize',12)
            set(gcf,'Position',[100 100 300 300])

        % zoom in
        figure(fuq)
            xlim([tstep-0.005 2.03])
            ylim([1.0295 1.035])

        figure(fud)
            xlim([tstep-0.005 2.03])
            ylim([0.042 0.047])
%             xlim([2.015 2.07])
%            ylim([0.04475 0.04555])

%% voltages and currents in loads

   data_PSCAD   = data_PSCAD_iz_berg;
   out_lin      = out_lin_rl;
   id_case      = '4_IZ_BERG';
   leg_nolin     = 'IZ/BERG';
   leg_lin       = 'RL';

   % Bus 5 (real 6): voltage and current 
   nbus = 6;
   nload = 3;
   nbus_real = 8;
    
   % correct time to start at zero
   Tinf = 0;
   Tsup = 0.2;
   time_nl = data_PSCAD.time - (Tstep-tstep_lin); 

   % linearization point from PSCAD
   vq0 = data_PSCAD.Uq8(find(data_PSCAD.time>=tstep-0.01,1));
   vd0 = data_PSCAD.Ud8(find(data_PSCAD.time>=tstep-0.01,1));
   Load_iq0 = data_PSCAD.Iq8(find(data_PSCAD.time>=tstep-0.01,1));
   Load_id0 = data_PSCAD.Id8(find(data_PSCAD.time>=tstep-0.01,1));

   vnq_lin     = out_lin.NET_vn6q + vq0*ones(nSamples,1);
   vnd_lin     = out_lin.NET_vn6d + vd0*ones(nSamples,1);   
   Load_iq_lin = -out_lin.Load3_iq + Load_iq0*ones(nSamples,1); 
   Load_id_lin = -out_lin.Load3_id + Load_id0*ones(nSamples,1); 

    f_vi = figure;

    figure(f_vi)
        
        ax_uq = subplot(2,2,1); 
            plot(time,vnq_lin,'LineWidth',1.3,'Color',color_lin{1})
            hold on
            plot(time_nl,data_PSCAD.Uq8,'LineWidth',1.3,'Color',color_nolin{1})

            grid on
            xlim([Tinf Tsup])
            set(gcf,'color','w');
            set(gca,'xticklabel',{[]},'FontName','Times New Roman','FontSize',11,'Position',pos221)
            ylabel(['$v_{' num2str(nbus_real) '}^q$ [pu]'],'Interpreter','latex','FontAngle','normal','FontWeight','bold','FontSize',12)
            handleLeg = legend(['lin-' leg_lin],['nolin-' leg_nolin],'Location','best','FontSize',8);
            handleLeg.ItemTokenSize = [15,1];

        ax_ud = subplot(2,2,3); 
            plot(time,vnd_lin,'LineWidth',1.3,'Color',color_lin{1})
            hold on
            plot(time_nl,data_PSCAD.Ud8,'LineWidth',1.3,'Color',color_nolin{1})

            grid on
            xlim([Tinf Tsup])
            set(gcf,'color','w');
            xlabel('time [s]','Interpreter','latex','FontName','Times New Roman','FontAngle','normal','FontWeight','bold','FontSize',12)
            set(gca,'FontName','Times New Roman','FontSize',11,'Position',pos223)
            ylabel(['$v_{' num2str(nbus_real) '}^d$ [pu]'],'Interpreter','latex','FontAngle','normal','FontWeight','bold','FontSize',12)
            

        ax_iq = subplot(2,2,2); 
            plot(time,Load_iq_lin,'LineWidth',1.3,'Color',color_lin{1})
            hold on
            plot(time_nl,data_PSCAD.Iq8,'LineWidth',1.3,'Color',color_nolin{1})

            grid on
            xlim([Tinf Tsup])
            set(gcf,'color','w');
            set(gca,'xticklabel',{[]},'FontName','Times New Roman','FontSize',11,'Position',pos222)
            ylabel(['$i_{L' num2str(nbus_real) '}^q$ [pu]'],'Interpreter','latex','FontAngle','normal','FontWeight','bold','FontSize',12)

        ax_id = subplot(2,2,4); 
            plot(time,Load_id_lin,'LineWidth',1.3,'Color',color_lin{1})
            hold on
            plot(time_nl,data_PSCAD.Id8,'LineWidth',1.3,'Color',color_nolin{1})

            grid on
            xlim([Tinf Tsup])
            set(gcf,'color','w');
            xlabel('time [s]','Interpreter','latex','FontName','Times New Roman','FontAngle','normal','FontWeight','bold','FontSize',11)
            set(gca,'FontName','Times New Roman','FontSize',11,'Position',pos224)
            ylabel(['$i_{L' num2str(nbus_real) '}^d$ [pu]'],'Interpreter','latex','FontAngle','normal','FontWeight','bold','FontSize',12)
    
            set(gcf,'Position',[100 100 540 340])
            exportgraphics(f_vi,[path_results id_case '_bus' num2str(nbus_real) '.emf'])
            exportgraphics(f_vi,[path_results id_case '_bus' num2str(nbus_real) '.eps'])

 % zoom in
        xlim(ax_uq,[tstep_lin-0.005 0.08])
        xlim(ax_ud,[tstep_lin-0.005 0.08])
        set(handleLeg,'visible','off')

%       ylim(ax_uq,[1.0165 1.0185])
%       ylim(ax_ud,[-0.0223 -0.02])
        ylim(ax_uq,[1.0167 1.0187])
        ylim(ax_ud,[-0.0230 -0.0207])

        xlim(ax_iq,[tstep_lin-0.005 0.08])
        xlim(ax_id,[tstep_lin-0.005 0.08])
        
%       ylim(ax_iq,[1.02325 1.02625])
%       ylim(ax_id,[0.3335 0.3355])
%       ylim(ax_iq,[1.0243 1.0265])
%       ylim(ax_id,[0.3338 0.3358])
        ylim(ax_iq,[1.0077 1.0120])
        ylim(ax_id,[0.3333 0.3357])

         exportgraphics(f_vi,[path_results id_case '_bus' num2str(nbus_real) '_zoom.emf'])
         exportgraphics(f_vi,[path_results id_case '_bus' num2str(nbus_real) '_zoom.eps'])

 %% voltages and currents in loads (MULTIPLE PLOTS linear)

   out_lin_array = {out_lin_iz,out_lin_rl};
   data_PSCAD   = data_PSCAD_iz_berg;
   %out_lin      = out_lin_rl;
   id_case      = '3_RL_IZ';
   leg_nolin     = 'IZ/PI';
   leg_lin       = 'RL';

   % Bus 5 (real 6): voltage and current 
   nbus = 6;
   nload = 3;
   nbus_real = 8;
    
   % correct time to start at zero
   Tinf = 0;
   Tsup = 0.2;
   time_nl = data_PSCAD.time - (Tstep-tstep_lin); 

   f_vi  = figure;
   ax_uq = subplot(2,2,1); 
   ax_ud = subplot(2,2,3); 
   ax_iq = subplot(2,2,2); 
   ax_id = subplot(2,2,4); 

   for idx = 1:2
       out_lin = out_lin_array{idx};          

       % linearization point from PSCAD
       vq0 = data_PSCAD.Uq8(find(data_PSCAD.time>=tstep-0.01,1));
       vd0 = data_PSCAD.Ud8(find(data_PSCAD.time>=tstep-0.01,1));
       Load_iq0 = data_PSCAD.Iq8(find(data_PSCAD.time>=tstep-0.01,1));
       Load_id0 = data_PSCAD.Id8(find(data_PSCAD.time>=tstep-0.01,1));
    
       vnq_lin     = out_lin.NET_vn6q + vq0*ones(nSamples,1);
       vnd_lin     = out_lin.NET_vn6d + vd0*ones(nSamples,1);   
       Load_iq_lin = -out_lin.Load3_iq + Load_iq0*ones(nSamples,1); 
       Load_id_lin = -out_lin.Load3_id + Load_id0*ones(nSamples,1); 


    figure(f_vi)        
        ax_uq = subplot(2,2,1); 
            hold on
            plot(time,vnq_lin,'LineWidth',1.3,'Color',color_lin{idx})

        ax_ud = subplot(2,2,3); 
            hold on
            plot(time,vnd_lin,'LineWidth',1.3,'Color',color_lin{idx})    

        ax_iq = subplot(2,2,2); 
            hold on
            plot(time,Load_iq_lin,'LineWidth',1.3,'Color',color_lin{idx})

        ax_id = subplot(2,2,4); 
            hold on
            plot(time,Load_id_lin,'LineWidth',1.3,'Color',color_lin{idx})              
   end

    figure(f_vi)        
        ax_uq = subplot(2,2,1); 
            hold on
            plot(time_nl,data_PSCAD.Uq8,'LineWidth',1.3,'Color',color_nolin{1})

            grid on
            xlim([Tinf Tsup])
            set(gcf,'color','w');
            set(gca,'xticklabel',{[]},'FontName','Times New Roman','FontSize',11,'Position',pos221)
            ylabel(['$v_{' num2str(nbus_real) '}^q$ [pu]'],'Interpreter','latex','FontAngle','normal','FontWeight','bold','FontSize',12)

        ax_ud = subplot(2,2,3); 
            hold on
            plot(time_nl,data_PSCAD.Ud8,'LineWidth',1.3,'Color',color_nolin{1})

            grid on
            xlim([Tinf Tsup])
            set(gcf,'color','w');
            xlabel('time [s]','Interpreter','latex','FontName','Times New Roman','FontAngle','normal','FontWeight','bold','FontSize',12)
            set(gca,'FontName','Times New Roman','FontSize',11,'Position',pos223)
            ylabel(['$v_{' num2str(nbus_real) '}^d$ [pu]'],'Interpreter','latex','FontAngle','normal','FontWeight','bold','FontSize',12)
            

        ax_iq = subplot(2,2,2); 
            hold on
            plot(time_nl,data_PSCAD.Iq8,'LineWidth',1.3,'Color',color_nolin{1})

            grid on
            xlim([Tinf Tsup])
            set(gcf,'color','w');
            set(gca,'xticklabel',{[]},'FontName','Times New Roman','FontSize',11,'Position',pos222)
            ylabel(['$i_{L' num2str(nbus_real) '}^q$ [pu]'],'Interpreter','latex','FontAngle','normal','FontWeight','bold','FontSize',12)

        ax_id = subplot(2,2,4); 
            hold on
            plot(time_nl,data_PSCAD.Id8,'LineWidth',1.3,'Color',color_nolin{1})

            grid on
            xlim([Tinf Tsup])
            set(gcf,'color','w');
            xlabel('time [s]','Interpreter','latex','FontName','Times New Roman','FontAngle','normal','FontWeight','bold','FontSize',11)
            set(gca,'FontName','Times New Roman','FontSize',11,'Position',pos224)
            ylabel(['$i_{L' num2str(nbus_real) '}^d$ [pu]'],'Interpreter','latex','FontAngle','normal','FontWeight','bold','FontSize',12)
    
            set(gcf,'Position',[100 100 540 340])           
            handleLeg = legend(ax_iq,['lin-' 'IZ'],['lin-' leg_lin],['nolin-' leg_nolin],'Location','best','FontSize',8);
            handleLeg.ItemTokenSize = [15,1];
            exportgraphics(f_vi,[path_results id_case '_bus' num2str(nbus_real) '.emf'])
            exportgraphics(f_vi,[path_results id_case '_bus' num2str(nbus_real) '.eps'])               

 % zoom in
        xlim(ax_uq,[tstep_lin-0.005 0.08])
        xlim(ax_ud,[tstep_lin-0.005 0.08])
        set(handleLeg,'visible','off')

%       ylim(ax_uq,[1.0165 1.0185])
%       ylim(ax_ud,[-0.0223 -0.02])

        xlim(ax_iq,[tstep_lin-0.005 0.08])
        xlim(ax_id,[tstep_lin-0.005 0.08])
        
%         ylim(ax_iq,[1.0077 1.0120])
%         ylim(ax_id,[0.3333 0.3357])

         exportgraphics(f_vi,[path_results id_case '_bus' num2str(nbus_real) '_zoom.emf'])
         exportgraphics(f_vi,[path_results id_case '_bus' num2str(nbus_real) '_zoom.eps'])

 %% voltages and currents in loads (MULTIPLE PLOTS nonlinear)

   out_lin_array = {out_lin_rl,out_lin_rl};
   id_case      = '5_RL_IZ';

   % Bus 5 (real 6): voltage and current 
   nbus = 6;
   nload = 3;
   nbus_real = 8;
    
   % correct time to start at zero
   Tinf = 0;
   Tsup = 0.2;
   time_nl = data_PSCAD.time - (Tstep-tstep_lin); 

   f_vi  = figure;
   ax_uq = subplot(2,2,1); 
   ax_ud = subplot(2,2,3); 
   ax_iq = subplot(2,2,2); 
   ax_id = subplot(2,2,4); 

   for idx = 1:2
       data_PSCAD   = data_PSCAD_array{idx};
       out_lin = out_lin_array{idx};          

       % linearization point from PSCAD
       vq0 = data_PSCAD.Uq8(find(data_PSCAD.time>=tstep-0.01,1));
       vd0 = data_PSCAD.Ud8(find(data_PSCAD.time>=tstep-0.01,1));
       Load_iq0 = data_PSCAD.Iq8(find(data_PSCAD.time>=tstep-0.01,1));
       Load_id0 = data_PSCAD.Id8(find(data_PSCAD.time>=tstep-0.01,1));
    
       vnq_lin     = out_lin.NET_vn6q + vq0*ones(nSamples,1);
       vnd_lin     = out_lin.NET_vn6d + vd0*ones(nSamples,1);   
       Load_iq_lin = -out_lin.Load3_iq + Load_iq0*ones(nSamples,1); 
       Load_id_lin = -out_lin.Load3_id + Load_id0*ones(nSamples,1); 


    figure(f_vi)        
        ax_uq = subplot(2,2,1); 
            hold on
            plot(time,vnq_lin,'LineWidth',1.3,'Color',color_lin{idx})
            hold on
            plot(time_nl,data_PSCAD.Uq8,'LineWidth',1.3,'Color',color_nolin{idx})

        ax_ud = subplot(2,2,3); 
            hold on
            plot(time,vnd_lin,'LineWidth',1.3,'Color',color_lin{idx})    
            hold on
            plot(time_nl,data_PSCAD.Ud8,'LineWidth',1.3,'Color',color_nolin{idx})

        ax_iq = subplot(2,2,2); 
            hold on
            plot(time,Load_iq_lin,'LineWidth',1.3,'Color',color_lin{idx})
            hold on
            plot(time_nl,data_PSCAD.Iq8,'LineWidth',1.3,'Color',color_nolin{idx})

        ax_id = subplot(2,2,4); 
            hold on
            plot(time,Load_id_lin,'LineWidth',1.3,'Color',color_lin{idx})      
            hold on
            plot(time_nl,data_PSCAD.Id8,'LineWidth',1.3,'Color',color_nolin{idx})
   end

    figure(f_vi)        
        ax_uq = subplot(2,2,1); 
            grid on
            xlim([Tinf Tsup])
            set(gcf,'color','w');
            set(gca,'xticklabel',{[]},'FontName','Times New Roman','FontSize',11,'Position',pos221)
            ylabel(['$v_{' num2str(nbus_real) '}^q$ [pu]'],'Interpreter','latex','FontAngle','normal','FontWeight','bold','FontSize',12)

        ax_ud = subplot(2,2,3); 
            grid on
            xlim([Tinf Tsup])
            set(gcf,'color','w');
            xlabel('time [s]','Interpreter','latex','FontName','Times New Roman','FontAngle','normal','FontWeight','bold','FontSize',12)
            set(gca,'FontName','Times New Roman','FontSize',11,'Position',pos223)
            ylabel(['$v_{' num2str(nbus_real) '}^d$ [pu]'],'Interpreter','latex','FontAngle','normal','FontWeight','bold','FontSize',12)
            

        ax_iq = subplot(2,2,2); 
            grid on
            xlim([Tinf Tsup])
            set(gcf,'color','w');
            set(gca,'xticklabel',{[]},'FontName','Times New Roman','FontSize',11,'Position',pos222)
            ylabel(['$i_{L' num2str(nbus_real) '}^q$ [pu]'],'Interpreter','latex','FontAngle','normal','FontWeight','bold','FontSize',12)

        ax_id = subplot(2,2,4); 
            grid on
            xlim([Tinf Tsup])
            set(gcf,'color','w');
            xlabel('time [s]','Interpreter','latex','FontName','Times New Roman','FontAngle','normal','FontWeight','bold','FontSize',11)
            set(gca,'FontName','Times New Roman','FontSize',11,'Position',pos224)
            ylabel(['$i_{L' num2str(nbus_real) '}^d$ [pu]'],'Interpreter','latex','FontAngle','normal','FontWeight','bold','FontSize',12)
    
            set(gcf,'Position',[100 100 540 340])           
            handleLeg = legend(ax_iq,['lin-' 'RL'],['nolin-' 'RL/BERG'],['lin-' 'RL'],['nolin-' 'IZ/BERG'],'Location','best','FontSize',8);
            handleLeg.ItemTokenSize = [15,1];
            exportgraphics(f_vi,[path_results id_case '_bus' num2str(nbus_real) '.emf'])
            exportgraphics(f_vi,[path_results id_case '_bus' num2str(nbus_real) '.eps'])               

 % zoom in
        xlim(ax_uq,[tstep_lin-0.005 0.08])
        xlim(ax_ud,[tstep_lin-0.005 0.08])
        set(handleLeg,'visible','off')

%       ylim(ax_uq,[1.0169 1.0181])
%       ylim(ax_ud,[-0.0233 -0.0195])

        xlim(ax_iq,[tstep_lin-0.005 0.08])
        xlim(ax_id,[tstep_lin-0.005 0.08])
        
%         ylim(ax_iq,[1.0052 1.0288])
%         ylim(ax_id,[0.3333 0.3357])

         exportgraphics(f_vi,[path_results id_case '_bus' num2str(nbus_real) '_zoom.emf'])
         exportgraphics(f_vi,[path_results id_case '_bus' num2str(nbus_real) '_zoom.eps'])

%% WTF PASSA EN EL BUS 4 ???? qd ???


   out_lin      = out_lin_rl;
   id_case      = '4_IZ_BERG';
   leg_nolin     = 'IZ/BERG';
   leg_lin       = 'RL';

   % Bus 5 (real 6): voltage and current 
   nbus = 1;
   nbus_real = 4;
    
   % correct time to start at zero
   Tinf = 0;
   Tsup = 0.2;
   time_nl = data_PSCAD.time - (Tstep-tstep_lin); 

   % linearization point from PSCAD
   vq0 = data_PSCAD.Uq4(find(data_PSCAD.time>=tstep-0.01,1));
   vd0 = data_PSCAD.Ud4(find(data_PSCAD.time>=tstep-0.01,1));

   vnq_lin     = out_lin.NET_vn1q + vq0*ones(nSamples,1);
   vnd_lin     = out_lin.NET_vn1d + vd0*ones(nSamples,1);   
   Load_iq_lin = -out_lin.Load3_iq + Load_iq0*ones(nSamples,1); 
   Load_id_lin = -out_lin.Load3_id + Load_id0*ones(nSamples,1); 

    f_vi = figure;

    figure(f_vi)
        
        ax_uq = subplot(2,2,1); 
            plot(time,vnq_lin,'LineWidth',1.3,'Color',color_lin{1})
            hold on
            plot(time_nl,data_PSCAD.Uq4,'LineWidth',1.3,'Color',color_nolin{1})

            grid on
            xlim([Tinf Tsup])
            set(gcf,'color','w');
            set(gca,'xticklabel',{[]},'FontName','Times New Roman','FontSize',11,'Position',pos221)
            ylabel(['$v_{' num2str(nbus_real) '}^q$ [pu]'],'Interpreter','latex','FontAngle','normal','FontWeight','bold','FontSize',12)
            handleLeg = legend(['lin-' leg_lin],['nolin-' leg_nolin],'Location','best','FontSize',8);
            handleLeg.ItemTokenSize = [15,1];

        ax_ud = subplot(2,2,3); 
            plot(time,vnd_lin,'LineWidth',1.3,'Color',color_lin{1})
            hold on
            plot(time_nl,data_PSCAD.Ud4,'LineWidth',1.3,'Color',color_nolin{1})

            grid on
            xlim([Tinf Tsup])
            set(gcf,'color','w');
            xlabel('time [s]','Interpreter','latex','FontName','Times New Roman','FontAngle','normal','FontWeight','bold','FontSize',12)
            set(gca,'FontName','Times New Roman','FontSize',11,'Position',pos223)
            ylabel(['$v_{' num2str(nbus_real) '}^d$ [pu]'],'Interpreter','latex','FontAngle','normal','FontWeight','bold','FontSize',12)
            

