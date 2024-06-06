function[sem] = SS_SEM(SG,we_0,is_q0,is_d0,ik1_q0,ik2_q0,if_d0,ik_d0,wbase,sg, NUMBER, sem_u, sem_y)
Rs = SG.Rs_pu(sg);
Rkd         = SG.R1d_pu(sg);
Rkq1        = SG.R1q_pu(sg);
Rkq2        = SG.R2q_pu(sg);
Rf_prime    = SG.Rf_pu(sg);
Lmd         = SG.Lmd_pu(sg);
Lmq         = SG.Lmq_pu(sg);
Ll          = SG.Ll_pu(sg);
Llkd        = SG.L1d_pu(sg);
Llkq1       = SG.L1q_pu(sg);
Llkq2       = SG.L2q_pu(sg);
Llfd_prime  = SG.Lfd_pu(sg);


Lmod   = [-(Lmd+Ll) Lmd Lmd 0 0 0;     
          -Lmd (Llkd+Lmd) Lmd 0 0 0; 
          -Lmd Lmd (Llfd_prime+Lmd) 0 0 0;
           0 0 0 -(Lmq+Ll) Lmq Lmq; 
           0 0 0 -Lmq (Llkq1+Lmq) Lmq; 
           0 0 0 -Lmq Lmq (Llkq2+Lmq)]/wbase;
LmodInv = inv(Lmod);
R8     = [-Rs 0 0 0 0 0;
           0 Rkd 0 0 0 0;
           0 0 (Rf_prime) 0 0 0;
           0 0 0 -Rs 0 0;
           0 0 0 0 Rkq1 0;
           0 0 0 0 0 Rkq2];
wLT    = [0 0 0 we_0*(Lmq+Ll) -we_0*Lmq -we_0*Lmq;
          0 0 0 0 0 0;
          0 0 0 0 0 0;
         -we_0*(Lmd+Ll) we_0*Lmd we_0*Lmd 0 0 0;
          0 0 0 0 0 0;
          0 0 0 0 0 0];
IL1    = [1 0 0 0 0 0 -(is_q0*(Lmq+Ll)-Lmq*ik1_q0 -Lmq*ik2_q0);
          0 1 0 0 0 0 0;
          0 0 1 0 0 0 0;
          0 0 0 1 0 0 -(-is_d0*(Lmd+Ll)+Lmd*ik_d0+Lmd*if_d0);
          0 0 0 0 1 0 0;
          0 0 0 0 0 1 0];
Asem    = -LmodInv*(R8+wLT);
Bsem    = LmodInv*IL1; 
Csem    = [1 0 0 0 0 0;
           0 1 0 0 0 0;
           0 0 1 0 0 0;
           0 0 0 1 0 0;
           0 0 0 0 1 0;
           0 0 0 0 0 1;
           3/2*(is_q0*(-Lmd+Lmq)-Lmq*(ik1_q0 +ik2_q0)) 3/2*Lmd*is_q0 3/2*Lmd*is_q0 3/2*(is_d0*(-Lmd+Lmq)+Lmd*(ik_d0+if_d0)) -3/2*Lmq*is_d0 -3/2*Lmq*is_d0];
Csem(7,:) = Csem(7,:)*2/3; 
Dsem      = zeros(7,7);

for NUM=NUMBER:NUMBER
xisd     = sprintf('SG%d.xis_d',NUM);
xikd     = sprintf('SG%d.xik_d',NUM);
xifd     = sprintf('SG%d.xif_d',NUM);
xisq     = sprintf('SG%d.xis_q',NUM);
xik1q    = sprintf('SG%d.xik1_q',NUM);
xik2q    = sprintf('SG%d.xik2_q',NUM);
end

sem_x = {xisd  xikd  xifd  xisq  xik1q  xik2q}; 
sem   = ss(Asem,Bsem,Csem,Dsem,'StateName',sem_x,'inputname',sem_u,'outputname',sem_y);
end