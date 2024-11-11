function [absqq,absqd,absdq,absdd,angleqq,angleqd,angledq,angledd,GRID_matrix_qd] = ZQD_GRIDMATRIX(ss)
    syms s
    FSUP = 3e3;
    FINF = -3e3;
    STEP = 1;
    n      = 0;
    SAMPLES = ceil((FSUP-FINF)/STEP)+1;                                                  % 1 EIG per 1 Hz             
    karray  = linspace(FINF,FSUP,SAMPLES)*2*pi(); 

    s       = 1i*karray;

    [GRID_matrix_qd] = freqresp(ss,s);


    for w=1:1:size(karray,2)
        %Magnitude
        absqq(w) = abs(GRID_matrix_qd(1,1,w));
        absqd(w) = abs(GRID_matrix_qd(1,2,w));
        absdq(w) = abs(GRID_matrix_qd(2,1,w));
        absdd(w) = abs(GRID_matrix_qd(2,2,w));
        %Phase
        angleqq(w) = angle(GRID_matrix_qd(1,1,w));
        angleqd(w) = angle(GRID_matrix_qd(1,2,w));
        angledq(w) = angle(GRID_matrix_qd(2,1,w));
        angledd(w) = angle(GRID_matrix_qd(2,2,w));
    end
end