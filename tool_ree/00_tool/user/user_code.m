%% BASE VALUES
Vb_kV  = T_global.Vb_kV{1};
Sb_MVA = T_global.Sb_MVA{1};
Sb     = Sb_MVA*1e6;
Zb     = (Vb_kV*1e3)^2/(Sb);
fb     = T_global.f_Hz{1};
wb     = 2*pi*fb;
%Lb     = Zb/(2*pi*fb); %w in pu

%% CASE ARRANGEMENTS

% % THEVENINS
% % slack bus
%   delta_slk = 0;
% % %T_TH(:,:)=[];
% % L_real = 0.1;
% % L_pu_01 = L_real/Lb; %base 138kV
% % Lb_02   = (345*1e3)^2/(100*1e6)/(2*pi*fb);
% % L_pu_02 = L_real/Lb_02; %base 345kV


% LOADS: get R,X from P,Q (rated) for IEEE-9
% T_load{1,"X"} = ((230e3)^2/(Traw.load.QL(1)*1e6))/Zb;
% T_load{2,"X"} = ((230e3)^2/(Traw.load.QL(2)*1e6))/Zb;
% T_load{3,"X"} = ((230e3)^2/(Traw.load.QL(3)*1e6))/Zb;

% Calculate R,X loads in case of PQ loads

% for idx = 1:height(T_load)
%     bus = T_load{idx,"bus"};
%     T_load.V(idx)     = results.bus{results.bus.bus == bus,"Vm"};
%     T.load.theta(idx) = results.bus{results.bus.bus == bus,"theta"};
%     T_load.R(idx) = abs(results.bus.Vm(results.bus.bus==bus)^2/T.load.P(idx));
%     T_load.X(idx) = abs(results.bus.Vm(results.bus.bus==bus)^2/T.load.Q(idx));
% end

%T_user.Sb=T_user.Sb*5;

%% CASE DEFINITION

sheets = sheetnames(excel);

if any(contains(sheets,'CASE'))
##    T_case          = readtable(excel,'Sheet','CASE', 'NumHeaderLines', 0, 'ReadVariableNames', true, 'VariableNamingRule','preserve');        %case definition

    T_case          = read_sheet_to_struct(excel,'PF')
    if ~isempty(T_case)
        for idx_user = 1:height(T_user)
            bus = T_user.bus(idx_user);
            elementName = T_user.element{idx_user};

            Pbus = T_case.P(T_case.bus == bus);
            Qbus = T_case.Q(T_case.bus == bus);
            alfa = T_case{T_case.bus == bus,elementName};

            P_case = Pbus*alfa;
            Q_case = Qbus*alfa;
            V_case = results.bus.Vm(results.bus.bus == bus);
            delta_case = results.bus.theta(results.bus.bus == bus);
            Sb_case = T_user.Sb(idx_user)*alfa;

            T_user{T_user.number==idx_user,"P"} = P_case;
            T_user{T_user.number==idx_user,"Q"} = Q_case;
            T_user{T_user.number==idx_user,"V"} = V_case;
            T_user{T_user.number==idx_user,"delta"} = delta_case;
            T_user{T_user.number==idx_user,"Sb"} = Sb_case;
        end
    end
end

T_user= T_user(T_user{:,'Sb'}~=0,:);

%% READ POD data

if any(contains(sheets,'POD_data'))
    T_pod         = readtable(excel,'Sheet','POD_data');        %case definition
end

%% CALCULATE delta_slk

% SG base values
    id_slk = find(T_user.type==0); % Find slack
    REF_w  = 'REF_w'; %variable name for w_slack (ref)

    if ~isempty(id_slk)
        % SG/VSC as reference
            num_slk     = T_user.number(id_slk);
            bus_slk     = T_user.bus(id_slk);
            T_XX        = T_user(id_slk,:);
            elementName = T_XX.element{:};

        % Element base
        Vb_kV  = T_XX.Vb;       % rated RMS L-L
        Vb     = Vb_kV*1e3;            % rated RMS L-L
        Zb     = (Vb)^2/(Sb);

           switch elementName
                case 'SG'
                    run sg_slack.m
                case 'GFOR'
                    run gfor_slack.m
                case 'GFOL'
            end
    else
        % Thevenin as reference
            num_slk = 0;
            delta_slk = 0;
            bus_slk = 0;
    end
%delta_slk = 0.453117201293350-pi/3+pi/2
