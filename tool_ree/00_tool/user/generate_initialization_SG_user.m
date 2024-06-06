function [init_SG, initMachineBlocks] =  generate_initialization_SG_user(T_SG,results) 
    
    init_SG = cell(1,height(T_SG));
    initMachineBlocks = cell(height(T_SG),12);

    for sg = 1:1:height(T_SG)  
        %% Data from the power-flow ---------------------------------------
        bus = T_SG.bus(sg);                
        num      = T_SG.number(sg);
        theta0   = results.global.theta(bus)*pi/180;
        V        = results.global.Vm(bus)/sqrt(3)*sqrt(2);   % PCC line voltage RMS
        Psg0     = results.user.P(num)*T_SG.Sb(sg)/T_SG.Sn(sg);
        Qsg0     = results.user.Q(num)*T_SG.Sb(sg)/T_SG.Sn(sg);
        
        %SG current
        I = abs(conj((Psg0+1i*Qsg0)/V));
        phi = -acos(Psg0./(sqrt(Psg0.^2+Qsg0.^2))).*sign(Qsg0./Psg0) + theta0;
        
        % Internal voltage
        E = (V*cos(theta0)+1i*V*sin(theta0))+(T_SG.Rs_pu(sg)+1i*T_SG.Xq(sg))*(I*cos(phi)+1i*I*sin(phi));
        Eq = real(E);
        Ed = imag(E);
        
        Emag = abs(E);
        delta = abs(atan(Ed/Eq)); %rotor angle
        
        % qd currents
        Iq = I*cos(-delta+phi);
        Id = -I*sin(-delta+phi);
        
        % qd terminal voltage
        Vq = V.*cos(theta0-delta);
        Vd = -V.*sin(theta0-delta);
        
        % qd terminal voltage (REF: NET)
        Vq_NET = V.*cos(theta0)*T_SG.Vbpubase(sg); %OJU AQUI ELS RMS (?)
        Vd_NET = -V.*sin(theta0)*T_SG.Vbpubase(sg);
        
        % Field voltage
        Eq_tr = Emag-Id*(T_SG.Xq(sg)-T_SG.Xd_tr(sg));
        Efd = Eq_tr + (T_SG.Xd(sg)-T_SG.Xd_tr(sg))*Id;
        Ifd = Efd/T_SG.Lmd_pu(sg);
       
        %% calculate non-linear model initial values ----------------------
        
        % Initial values for: Syncronous Machine pu Fundamental
        % Initial conditions [dw(%) th(deg) ia,ib,ic(pu) pha,phb,phc(deg) Vf(pu)]
        initSG.dw = 0;
        initSG.th = delta*180/pi-90;
        initSG.ia = I;
        initSG.ib = I;
        initSG.ic = I;
        initSG.pha = phi*180/pi;
        initSG.phb = initSG.pha - 120;
        initSG.phc = initSG.pha + 120;

        % Initial values for: Governor 
        initSG.Pm = Psg0+(Iq^2+Id^2)*T_SG.Rs_pu(sg);

         %Initial values for: Exciter
        initSG.Vf = Efd;
        initSG.Vref = initSG.Vf/T_SG.KA(sg)+V;    %unused (!)
        
        % 2 - Initialize Multi-mass shaft M2 to M5:
        Pm = initSG.Pm;
        % M1
        initSG.delta1 = initSG.th*pi/180;
        % M2
        initSG.delta2 = initSG.delta1 + Pm/T_SG.K12(sg);
        % M3
        initSG.delta3 = Pm*(1-T_SG.F2(sg))/T_SG.K23(sg) + initSG.delta2;
        % M4
        initSG.delta4 = Pm*(1-T_SG.F2(sg)-T_SG.F3(sg))/T_SG.K34(sg) + initSG.delta3;
        % M5
        initSG.delta5 = Pm*(1-T_SG.F2(sg)-T_SG.F3(sg)-T_SG.F4(sg))/T_SG.K45(sg) + initSG.delta4;
            
        %% initialize simulink blocks
        
        %initMachineBlocks{sg,1} = [T_SG.Sn(sg)/T_Global.Sb 1  T_SG.fn(sg)];                    %grid in RMS,  L-L
        initMachineBlocks{sg,1} = [T_SG.Sn(sg)/T_SG.Sb(sg) sqrt(3)/sqrt(2)  T_SG.fn(sg)];       %grid in peak, F-L

        initMachineBlocks{sg,2} = [T_SG.Rs_pu(sg)  T_SG.Ll_pu(sg)  T_SG.Lmd_pu(sg)  T_SG.Lmq_pu(sg)];
        initMachineBlocks{sg,3} = [T_SG.Rf_pu(sg)  T_SG.Lfd_pu(sg)];
        initMachineBlocks{sg,4} = [T_SG.R1d_pu(sg)  T_SG.L1d_pu(sg)  T_SG.R1q_pu(sg)   T_SG.L1q_pu(sg)  T_SG.R2q_pu(sg)  T_SG.L2q_pu(sg)];

        % single-mass
        if (T_SG.genType{sg} == 0)
            initMachineBlocks{sg,5} = [T_SG.H(sg)  T_SG.D1(sg) 1];
         % multi-mass
        elseif (T_SG.genType{sg} == 1) 
            initMachineBlocks{sg,5} = [T_SG.H1(sg)  T_SG.D1(sg) 1];
        end

        initMachineBlocks{sg,6} = [initSG.dw initSG.th initSG.ia initSG.ib initSG.ic initSG.pha initSG.phb initSG.phc initSG.Vf];

        %% Multimass parameters and initialization ------------------------
        initMachineBlocks{sg,7} = [initSG.Pm initSG.th];
        initMachineBlocks{sg,8} = [T_SG.T2(sg)    T_SG.T3(sg)   T_SG.T4(sg)   T_SG.T5(sg)];
        initMachineBlocks{sg,9} = [T_SG.F2(sg)    T_SG.F3(sg)   T_SG.F4(sg)   T_SG.F5(sg)];
        initMachineBlocks{sg,10} = [T_SG.H2(sg)    T_SG.H3(sg)   T_SG.H4(sg)   T_SG.H5(sg)];
        initMachineBlocks{sg,11} = [T_SG.K12(sg)   T_SG.K23(sg)  T_SG.K34(sg)  T_SG.K45(sg)];
        initMachineBlocks{sg,12} = [T_SG.D2(sg)    T_SG.D3(sg)   T_SG.D4(sg)   T_SG.D5(sg)];
        
        init_SG{sg} = initSG;
    end
end
