function lbd = PLOT_BODE(ss_zsystem,FINF,FSUP,STEP)
%%
%%  v1
%%

syms s
n      = 0;

%%      FREQUENCY

SAMPLES = ceil((FSUP-FINF)/STEP)+1;                                                  % 1 EIG per 1 Hz             
karray  = linspace(FINF,FSUP,SAMPLES)*2*pi(); 
karray(1181) = karray(1180);
%karray(1199) = karray(1198);

s       = 1i*karray;

lbd = {}; %store eig

fwb = waitbar(0.1,'Loading model');

[GRID_matrix] = freqresp(ss_zsystem,s);
[GROWS,GCOLUMNS]    = size(GRID_matrix);

waitbar(0.85,fwb,'Sourcer loaded'); 
waitbar(0.9,fwb,'Calculating eigenvalues');    

for k =FINF*2*pi:STEP*2*pi:FSUP*2*pi                                       %    Frequency                                     
    n     = n+1;
        
%%  GNC
  
    G_matrix    = GRID_matrix(1:GROWS,(n*GROWS-(GROWS-1)):n*GROWS);
    
    L_cl   = G_matrix;       

    [lambda] = diag(eig(L_cl)); 
    lbd{n} = lambda;
    
    
    nl       = length(lambda);
    fr(n)    = k/(2*pi);
        

%%  ORDER EIGS
    ERROR=1;
    FEIGFRORDER
end
       
%%  PLOT BODE

FCLFORMAT_BODE
    
%
waitbar(1,fwb,'Complete');
close (fwb)
%
end

