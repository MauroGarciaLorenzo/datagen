function [T_SG] =  generate_parameters_SG(T_SG) 

% rename table columns for simplicity
if sum(strcmp('Sb',T_SG.Properties.VariableNames)) == 0
    T_SG.Properties.VariableNames{'Sb_MVA'} = 'Sb';
    T_SG.Properties.VariableNames{'Vb_ll_rms'} = 'Vb';
end
 
T_SG.Zb = T_SG.Vb.^2./(T_SG.Sb*1e6);

    for sg = 1:1:height(T_SG)
        %% SG electrical & mechanical parameters --------------------------
        T_SG.wn(sg) = 2*pi*T_SG.fn(sg);
        
        Xl = T_SG.Xl(sg);
        Xd = T_SG.Xd(sg);
        Xd_tr = T_SG.Xd_tr(sg);
        Xd_subtr = T_SG.Xd_subtr(sg);
        Xq = T_SG.Xq(sg);
        Xq_tr = T_SG.Xq_tr(sg);
        Xq_subtr = T_SG.Xq_subtr(sg);
        Tdo_tr = T_SG.Tdo_tr(sg);
        Tdo_subtr = T_SG.Tdo_subtr(sg);
        Tqo_tr = T_SG.Tqo_tr(sg);
        Tqo_subtr = T_SG.Tqo_subtr(sg);
        wn = T_SG.wn(sg);
       
        % Conversion from standard to equivalent circuit parameters
        Ll_pu = Xl; 
        Lmd_pu = Xd - Xl; 
        Lmq_pu = Xq - Xl; 

        Lfd_pu = (Lmd_pu*(Xd_tr-Xl))/(Lmd_pu-Xd_tr+Xl);
        L1q_pu = (Lmq_pu*(Xq_tr-Xl))/(Lmq_pu-Xq_tr+Xl);

        L1d_pu = (Xd_subtr-Xl)*(Lmd_pu*Lfd_pu)/(Lmd_pu*Lfd_pu-(Lfd_pu+Lmd_pu)*(Xd_subtr-Xl));
        L2q_pu = (Xq_subtr-Xl)*(Lmq_pu*L1q_pu)/(Lmq_pu*L1q_pu-(L1q_pu+Lmq_pu)*(Xq_subtr-Xl));

        Rf_pu = (Lmd_pu+Lfd_pu)/(Tdo_tr*wn);
        R1d_pu = 1/(Tdo_subtr*wn)*(L1d_pu+Lmd_pu*Lfd_pu/(Lmd_pu+Lfd_pu));

        R1q_pu = (Lmq_pu+L1q_pu)/(Tqo_tr*wn);
        R2q_pu = 1/(Tqo_subtr*wn)*(L2q_pu+Lmq_pu*L1q_pu/(Lmq_pu+L1q_pu));
        
        % save to SG
        T_SG.Ll_pu(sg) = Ll_pu; 
        T_SG.Lmd_pu(sg) = Lmd_pu; 
        T_SG.Lmq_pu(sg) = Lmq_pu; 
        T_SG.Lfd_pu(sg) = Lfd_pu;
        T_SG.L1q_pu(sg) = L1q_pu;
        T_SG.L1d_pu(sg) = L1d_pu;
        T_SG.L2q_pu(sg) = L2q_pu;
        T_SG.Rf_pu(sg) = Rf_pu;
        T_SG.R1d_pu(sg) = R1d_pu;
        T_SG.R1q_pu(sg) = R1q_pu;
        T_SG.R2q_pu(sg) = R2q_pu;
        
        if T_SG.genType{sg} == 'S'
            T_SG.genType{sg} = 0;
        elseif T_SG.genType{sg} == 'M'
            T_SG.genType{sg} = 1; %multimass
        end
        
        % BASE
        Sbpu     = T_SG.Sn(sg)/T_SG.Sb(sg);
        %Vbpu     = 1;
        T_SG.Vbpubase(sg) = 1; %Vbpu/sqrt(3)*sqrt(2);
        T_SG.Ibpubase(sg) = Sbpu*2/3; %Sbpu/Vbpu/sqrt(3)*sqrt(2);
        
        T_SG.Rsnb(sg) = T_SG.Vn(sg)^2/(T_SG.Sn(sg)*0.02)/T_SG.Zb(sg);     %Snubber resistance
        
    end
end
