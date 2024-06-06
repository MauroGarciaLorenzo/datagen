%% SG as slack

%% Parameters

    % Element base
    Vb_kV  = T_XX.Vb;       % rated RMS L-L, kV
    Vb     = Vb_kV*1e3;     % rated RMS L-L   
    Ssg    = T_XX.Sb*1e6;      % SG rated power  
    run SG_param_tool.m

%% Delta slack

    %Synchronous generator
    % Psg0 = -results.gen(ng,2)./(Ssg/1e6);  %Results from power flow 
    % Qsg0 = -results.gen(ng,3)./(Ssg/1e6);
    % V = results.bus(ng,8);
    % delta_bus = 0;
    
    Psg0 = T_XX.P*(Sb/Ssg);
    Qsg0 = T_XX.Q*(Sb/Ssg);
    V = T_XX.V;
    delta0 = T_XX.delta*pi/180; 
    
    
    % SG terminals voltage
    Itr = conj((Psg0+1i*Qsg0)./V);
    Vin = V+Itr*(Rtr+1i*Xtr);
    theta_in = atan(imag(Vin)/real(Vin));
    
    % Snubber current
    Isnb = Vin/(Rsnb);
    
    %SG current
    Isg = Isnb+Itr;
    I = abs(Isg);
    
    % Aparent power (inside the transformer)
    Sin = Vin*conj(Isg);
    Pin = real(Sin);
    Qin = imag(Sin);
    phi = -acos(Pin./(sqrt(Pin.^2+Qin.^2))).*sign(Qin./Pin);
    
    % Internal voltage
    E = abs(Vin)+(Rs_pu+1i*Xq)*(I.*cos(phi)+1i*I.*sin(phi));
    Eq = real(E);
    Ed = imag(E);
    
    Emag = abs(E);
    delta = atan(Ed./Eq); %rotor angle
    delta_slk = delta + theta_in; %rotor angle
