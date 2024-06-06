#! python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Power Systems Computer Aided Design (PSCAD)
# ------------------------------------------------------------------------------
#  This Python script is used to externally control PSCAD for the purpose of
#  testing and batch control.
#
#     George Wai
#     PSCAD Support Team <support@pscad.com>
#     Manitoba Hydro International
#     Winnipeg, Manitoba. CANADA
# 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import sys, os
import mhrc.automation
import math
import cmath

# Imports function readraw
from read_RAW_REE_v1 import readraw

# Import other utilities to perform cool stuff
from win32com.client.gencache import EnsureDispatch as Dispatch

import win32com.client
import shutil

#---------------------------------------------------------------------
# Configuration
#---------------------------------------------------------------------
pscad_version = 'PSCAD 4.6.2 (x64)'
fortran_version = 'GFortran 4.6.2'
fortran_ext = '.gf46'
project_name = 'IEEE_118_REE_GFOR_v2'

# Working directory
working_dir = os.getcwd() + "\\"

#---------------------------------------------------------------------
# Main script 
#---------------------------------------------------------------------

#----  Read .RAW file from PSSE --------
# In this example, we prepared to scenarios: Winter (IEEE118busREE_Winter.raw) and Summer (IEEE118busREE_Summer.raw)
rawfilename = 'IEEE118busREE_Winter.raw'    # .RAW file name from PSSE (include .raw in the name)

# The function 'readraw(rawfilename)' reads the .RAW file from PSSE and assign each value to a variable
# To understand the nomenclature used for each variable, please consult the function 'readraw(rawfilename)'
# Note: several variables are loaded but not used. However, we still load them to show the capability of the function
sbase,freqbase,busindex,bus_name,nominal_voltage,bus_type,voltage,angle,ZloadP,ZloadQ,IloadP,IloadQ,SloadP,SloadQ,load_buses,qsh,genp,genq,genbasmva,gennodeindex,generation_buses,genqmax,genqmin,busnumber1,busnumber2,branchR,branchX,branchBsh = readraw(rawfilename)

gfol_buses = []
gfor_buses = [12,65, 72, 90,112]

#---------------------------------------

# Launch specific PSCAD and Fortran version
pscad = mhrc.automation.launch_pscad(pscad_version=pscad_version, fortran_version=fortran_version)

if pscad:
    try:
        # Check if Licensing Cetificate exists, on this computer
        #pro = os.path.isfile(r"C:\Users\Public\Documents\Manitoba HVDC Research Centre\Licensing\Licenses\74.xml")
        #edu = os.path.isfile(r"C:\Users\Public\Documents\Manitoba HVDC Research Centre\Licensing\Licenses\72.xml")
        # Existing certificate exists!, set PSCAD to use Certificate License
        #if(pro or edu):
        #    pscad.settings(cl_use_advanced='true')
        #    print("PSCAD is using a Certificate license!")
        # No Certificate exists, set PSCAD to use dongle License
        #else:
        #    pscad.settings(cl_use_advanced='false')
        #    print("PSCAD is using a Dongle license!")   

        # Load the project
        pscad.load([working_dir + project_name + ".pscx"])
        project = pscad.project(project_name) 
        project.focus()
        
        # Get the "Main" canvas
        main = project.user_canvas('Main')

        # Get the GENERATORS based on their IDs in PSCAD
        G12 = main.user_cmp(1995774957)
        G19 = main.user_cmp(1268372905)
        G31 = main.user_cmp(1572987196)
        G32 = main.user_cmp(867417209)
        G34 = main.user_cmp(934757788)
        G36 = main.user_cmp(851174825)
        G40 = main.user_cmp(1359153743)
        G46 = main.user_cmp(1484331182)
        G49 = main.user_cmp(548869790)
        G54 = main.user_cmp(1977860585)
        G59 = main.user_cmp(1941803138)
        G65 = main.user_cmp(1533293726)
        G70 = main.user_cmp(757427482)
        G72 = main.user_cmp(925156312)
        G73 = main.user_cmp(281376095)
        G76 = main.user_cmp(1768047330)
        G77 = main.user_cmp(109756332)
        G80 = main.user_cmp(2063965358)
        G85 = main.user_cmp(1295422624)
        G89 = main.user_cmp(213814524)
        G90 = main.user_cmp(1311657125)
        G92 = main.user_cmp(1453211248)
        G100 = main.user_cmp(1259949634)
        G105 = main.user_cmp(1993349017)
        G107 = main.user_cmp(1756073464)
        G110 = main.user_cmp(1873516057)
        G112 = main.user_cmp(1899630866)
        G113 = main.user_cmp(1697711211)

        # Get the LOADS based on their IDs in PSCAD
        L1 = main.user_cmp(355772535)
        L2 = main.user_cmp(2075889964)
        L3 = main.user_cmp(1633065292)
        L4 = main.user_cmp(795201815)
        L6 = main.user_cmp(1109694619)
        L7 = main.user_cmp(2122178572)
        L11 = main.user_cmp(1252912570)
        L12 = main.user_cmp(1389461992)
        L13 = main.user_cmp(196342107)
        L14 = main.user_cmp(1650697278)
        L15 = main.user_cmp(868426204)
        L16 = main.user_cmp(1489990127)
        L17 = main.user_cmp(1353161041)
        L18 = main.user_cmp(2029229682)
        L19 = main.user_cmp(111711533)
        L20 = main.user_cmp(1233680572)
        L21 = main.user_cmp(2070384856)
        L22 = main.user_cmp(1602944079)
        L23 = main.user_cmp(750196737)
        L27 = main.user_cmp(1425201041)
        L28 = main.user_cmp(246421149)
        L29 = main.user_cmp(1184384478)
        L31 = main.user_cmp(1036380826)
        L32 = main.user_cmp(155730662)
        L33 = main.user_cmp(1207308759)
        L34 = main.user_cmp(339788228)
        L35 = main.user_cmp(2057419306)
        L36 = main.user_cmp(52614765)
        L37 = main.user_cmp(2027391823)
        L40 = main.user_cmp(1587900320)
        L41 = main.user_cmp(724674084)
        L42 = main.user_cmp(1643162914)
        L43 = main.user_cmp(1422883626)
        L44 = main.user_cmp(841793057)
        L45 = main.user_cmp(334096758)
        L46 = main.user_cmp(784377285)
        L47 = main.user_cmp(1661145129)
        L48 = main.user_cmp(1295145509)
        L49 = main.user_cmp(1239723354)
        L50 = main.user_cmp(1595411523)
        L51 = main.user_cmp(663191740)
        L52 = main.user_cmp(281714932)
        L53 = main.user_cmp(1882770843)
        L54 = main.user_cmp(2052434655)
        L55 = main.user_cmp(672158598)
        L56 = main.user_cmp(1292749812)
        L57 = main.user_cmp(1656013566)
        L58 = main.user_cmp(1141829721)
        L59 = main.user_cmp(1715701037)
        L60 = main.user_cmp(1407099651)
        L62 = main.user_cmp(1506365787)
        L66 = main.user_cmp(187273141)
        L67 = main.user_cmp(1542646664)
        L70 = main.user_cmp(445993164)
        L74 = main.user_cmp(340166427)
        L75 = main.user_cmp(1556363992)
        L76 = main.user_cmp(380686251)
        L78 = main.user_cmp(587018117)
        L79 = main.user_cmp(1255053212)
        L80 = main.user_cmp(1338776106)
        L82 = main.user_cmp(1776474015)
        L83 = main.user_cmp(801678794)
        L84 = main.user_cmp(2083244299)
        L85 = main.user_cmp(1135726181)
        L86 = main.user_cmp(1068789139)
        L88 = main.user_cmp(248729589)
        L90 = main.user_cmp(1198085567)
        L92 = main.user_cmp(393039845)
        L93 = main.user_cmp(1658341626)
        L94 = main.user_cmp(1234354693)
        L95 = main.user_cmp(1350520637)
        L96 = main.user_cmp(649475732)
        L97 = main.user_cmp(2028460271)
        L98 = main.user_cmp(477178811)
        L100 = main.user_cmp(1810778008)
        L101 = main.user_cmp(741285747)
        L102 = main.user_cmp(768425134)
        L103 = main.user_cmp(1256849860)
        L104 = main.user_cmp(124818367)
        L105 = main.user_cmp(543320891)
        L106 = main.user_cmp(2016926034)
        L107 = main.user_cmp(2058629118)
        L108 = main.user_cmp(929174750)
        L109 = main.user_cmp(1694773332)
        L110 = main.user_cmp(133944379)
        L112 = main.user_cmp(1296305589)
        L114 = main.user_cmp(2090916712)
        L115 = main.user_cmp(1974723035)
        L117 = main.user_cmp(360653833)
        L118 = main.user_cmp(1650814979)
        L77 = main.user_cmp(179682814)

        # Get the VSC-GFOL based on their IDs in PSCAD
        GFOL12 = main.user_cmp(1753353045)
        GFOL65 = main.user_cmp(66760541)
        GFOL72 = main.user_cmp(680010821)
        GFOL90 = main.user_cmp(2000430454)
        GFOL112 = main.user_cmp(1947362445)

        # Get the VSC-GFOL based on their IDs in PSCAD
        GFOR12 = main.user_cmp(714890535)
        GFOR65 = main.user_cmp(64204908)
        GFOR72 = main.user_cmp(1473527416)
        GFOR90 = main.user_cmp(1813439187)
        GFOR112 = main.user_cmp(853068135)
        


        # ----- Start loops that send variables to PSCAD -----

        # begin GENERATORS loop
        for k in generation_buses:

            # The Sync Machine model in PSCAD requires informing the machine's base current instead of its nominal power
            # Thus, we calculate the nominal current based on the nominal power
            Inom = genbasmva[k]/(math.sqrt(3)*nominal_voltage[k])

            # If the generator has a step-up transformer, we can calculate the power before the transformer in order to
            # have the power after the transformer matching the PSSE power flow results.
            Rt = 0  # Step-up transformer resistance
            Xt = 0  # Step-up transformer reactance
            Zt = complex(Rt, Xt)    # Step-up transformer impedance Z = R + jX

            S0pu = complex(genp[k],genq[k])/genbasmva[k]             # Define S = P + jQ
            Vt0 = cmath.rect(voltage[k], angle[k]*math.pi/180)       # Define voltage as a complex number
            I0pu = S0pu/(math.sqrt(3)*Vt0); I0pu = I0pu.conjugate()  # Calculate current
            # With the current and the transformer impedance, we can calculate the voltage before the transformer
            Vin0 = Vt0 + I0pu*Zt*math.sqrt(3)     # Calculate the voltage before the transformer (internal voltage)
            Vin0mag = abs(Vin0)                                      # Takes the magnitude of the internal voltage
            Vin0ang = cmath.phase(Vin0)                              # Takes the angle of the internal voltage
            S0in = genbasmva[k]*math.sqrt(3)*Vin0*I0pu.conjugate()   # Calculate the power before the transformer
            P0in,Q0in = S0in.real,S0in.imag                          # Separate active and reactive power

            # ----------- Send the variables to each model in PSCAD -----------
            print("Sending parameters to " +"G" + str(k))
            exec("G" + str(k) + ".set_parameters(" + "P0=" + '"' + str(P0in) + '"' + ")")
            exec("G" + str(k) + ".set_parameters(" + "Q0=" + '"' + str(Q0in) + '"' + ")")
            exec("G" + str(k) + ".set_parameters(" + "Vt0=" + '"' + str(Vin0mag) + '"' + ")")
            exec("G" + str(k) + ".set_parameters(" + "theta0=" + '"' + str(Vin0ang) + '"' + ")")
            exec("G" + str(k) + ".set_parameters(" + "Inom=" + '"' + str(Inom) + '"' + ")")
            exec("G" + str(k) + ".set_parameters(" + "StrSG=" + '"' + str(genbasmva[k]) + '"' + ")")
        print("End of GENERATORS loop")
        # end of GENERATORS loop ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # begin LOADS loop --------------------------------------------------------------------------------------------
        for k in load_buses:
            # Note: as all loads where defined as IZ (constant current for active power and constant impedance for
            # reactive power, the code takes directly these field only from the PSSE .RAW file.
            # If other load characteristics are considered, all of them need to be assigned as we do below

            # Note: load powers are entered per phase in PSCAD, that's why we divide by 3
            # Note: PSSE convention for constant power and constant current load is positive for load absorbing power
            # but for constant impedance load, the convention is negative power for load absorbing power
            # As PSCAD uses the convention as positive power for load absorbing power for all ZIP loads, we put a
            # negative sign in the power of constant impedance load from PSSE before sending to PSCAD

            # ----------- Send the variables to each model in PSCAD -----------
            print("Sending parameters to " +"L" + str(k))
            exec("L" + str(k) + ".set_parameters(" + "PO=" + '"' + str(IloadP[k]/3) + '"' + ")")
            exec("L" + str(k) + ".set_parameters(" + "QO=" + '"' + str(-ZloadQ[k]/3) + '"' + ")")
        print("End of LOADS loop")
        # end of LOADS loop ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # begin GFOL loop --------------------------------------------------------------------------------------------
        for k in gfol_buses:

            #---------------------------------
            # CONVERTER FILTER
            #---------------------------------

            # base values
            base_omega = 2*math.pi*freqbase # in [rad/s]
            base_power = genbasmva[k] # in [MVA
            base_ACvoltage = nominal_voltage[k] # in [kV]
            base_ACimpedance = base_ACvoltage**2/base_power # in [ohm]
            base_DCvoltage = 1000 # in [kV]
            base_current = base_power/base_ACvoltage/math.sqrt(3)  # in kA (RMS value)
            # VSC AC filter parameters
            filter_res_pu = 0.005  # in [pu]
            filter_induct_pu = 0.15  # in [pu]

            filter_res = filter_res_pu*base_ACimpedance # in [ohm]
            filter_induct = filter_induct_pu*base_ACimpedance/base_omega  # in [H]

            #---------------------------------
            # CONVERTER CONTROL
            #---------------------------------
            #PLL
            ts=0.1              # Settling time
            xi_pll = 0.707      # Damping
            wn_pll = 4/(ts*xi_pll)
            kp_pll = 2*wn_pll*xi_pll/(base_ACvoltage*math.sqrt(2/3))
            tau_pll = 2*xi_pll/wn_pll
            ki_pll = kp_pll/tau_pll

            # P controller mode
            P_ctrl_mode = 1 # 1 = P control (+frequency control, depending on mp value)
                            # 0 = DC voltage control

            # Q controller mode
            Q_ctrl_mode = 1 # 1 = Q control (+voltage control, depending on mq value)
                            # 0 = open loop (direct Q reference)

            # current controller (kp and ki are calculated in the PSCAD model according to the parameters defined here)
            tau_c = 0.001  # time constant

            # Current saturation
            Imax = 1.1 #pu
            en_current_saturation = 1 # 1 = saturation enabled
                                      # 0 = saturation disabled

            # Active power control
            tau_P = 0.1 # s
            kp_P = tau_c/tau_P
            ki_P = 1/tau_P

            # Reactive power control
            tau_Q = 0.1 # s
            kp_Q = tau_c/tau_Q
            ki_Q = 1/tau_Q

            # Frequency droop control
            mp = 20 # mp = 1/Rp (Rp=5%)
            w_filtP = 50 # Hz

            # Voltage droop control
            mq = 50 # mq = 1/Rq (Rq=2%)
            w_filtQ = 50 # Hz

            # Initial values
            Pvsc0 = genp[k];                    # Active power from power flow in MW
            Qvsc0 = genq[k];                    # Reactive power from power flow in Mvar
            Vvsc0 = voltage[k]*base_ACvoltage;  # Voltage from power flow in kV
            e_theta_vsc0 = angle[k]*math.pi/180-math.pi/2; # Bus angle from power flow in  rad

            Svsc0 = complex(Pvsc0, Qvsc0)   # Define S = P + jQ
            Vpoc0 = Vvsc0/math.sqrt(3)      # RMS ph-n voltage at point of connection (POC) 
            Ic = Svsc0/(3*Vpoc0)            # Current flowing through converter L filter
            Ic = Ic.conjugate()
            ic_q0 = Ic.real*math.sqrt(2)    # q-axis converter current (used to initialize integrator of P control)
            ic_d0 = -Ic.imag*math.sqrt(2)   # d-axis converter current (used to initialize integrator of Q control)
            Vc = Vpoc0 + Ic*complex(filter_res, base_omega*filter_induct) # Calculate the voltage applied by the converter (internal voltage)
            vc_q0 = Vc.real*math.sqrt(2)    # q-axis converter voltage (used to initialize the converter as V source)
            vc_d0 = -Vc.imag*math.sqrt(2)   # d-axis converter voltage (used to initialize the converter as V source)

            # ----------- Send the variables to each model in PSCAD -----------
            print("Sending parameters to " +"GFOL" + str(k))
            exec("GFOL" + str(k) + ".set_parameters(f_1="+str(freqbase)+", Ab="+ str(base_power)+", Vb_ac="+str(base_ACvoltage)+", Vb_dc="+str(base_DCvoltage)+
                                                    ",tau_c_loop="+str(tau_c)+", R_cf="+str(filter_res)+", L_cf="+str(filter_induct)+
                                                    ",Imax_rms =" +str(Imax*base_current)+", limit_act ="+str(en_current_saturation)+
                                                    ",Kp_PLL =" +str(kp_pll)+", Ki_PLL ="+str(ki_pll)+
                                                    ",P_ctrl_mode =" +str(P_ctrl_mode) +", Q_ctrl_mode =" +str(Q_ctrl_mode)+
                                                    ",kp_activepower_PI =" +str(kp_P)+ ", ki_activepower_PI =" +str(ki_P)+
                                                    ",mp_droop =" +str(mp) +", wp_droop =" +str(w_filtP)+", mq_droop =" +str(mq) +", wq_droop =" +str(w_filtQ)+ 
                                                    ",tP1=100, tP2=100, tP3=100, tP4=100, tP5=100"+
                                                    ",Pref0=" +str(Pvsc0/base_power)+", Pref1=0, Pref2=0, Pref3=0, Pref4=0, Pref5=0"+
                                                    ",tQ1=100, tQ2=100, tQ3=100, tQ4=100"+
                                                    ",Qref0="+str(Qvsc0/base_power)+ ",Qref1=0, Qref2=0, Qref3=0, Qref4=0"+
                                                    ",v_ref="+str(voltage[k])+",theta0="+str(e_theta_vsc0)+",vc_q0="+str(vc_q0)+",vc_d0="+str(vc_d0)+",i_ref_q0="+str(-ic_q0)+",i_ref_d0="+str(-ic_d0)+")")                      


        # begin GFOL loop --------------------------------------------------------------------------------------------
        

for k in gfor_buses:

            #---------------------------------
            # CONVERTER FILTER
            #---------------------------------

            # base values
            base_omega = 2*math.pi*freqbase # in [rad/s]
            base_power = genbasmva[k] # in [MVA
            base_ACvoltage = nominal_voltage[k] # in [kV]
            base_ACimpedance = base_ACvoltage**2/base_power # in [ohm]
            base_DCvoltage = 1000 # in [kV]
            base_current = base_power/base_ACvoltage/math.sqrt(3)  # in kA (RMS value)

            # VSC AC filter parameters
            filter_res_pu = 0.005  # in [pu]
            filter_induct_pu = 0.15  # in [pu]
            filter_cap_pu = 0.15  # in [pu]

            filter_res = filter_res_pu*base_ACimpedance # in [ohm]
            filter_induct = filter_induct_pu*base_ACimpedance/base_omega  # in [H]
            filter_cap =  filter_cap_pu/base_ACimpedance/base_omega*1e6  # in [uF]
            filter_res_cap =1/(3*10*base_omega*filter_cap/1e6);   # in [ohm] - passive damping for LC filter

            # transformer parameters
            trafo_res_pu = 0.002  # in [pu]
            trafo_induct_pu = 0.1  # in [pu]

            trafo_res = trafo_res_pu*base_ACimpedance # in [ohm]
            trafo_induct = trafo_induct_pu*base_ACimpedance/base_omega  # in [H]

            #---------------------------------
            # CONVERTER CONTROL
            #---------------------------------
            # controller mode
            PQ_control = 1  # 0: No PQ control, 1: PQ-Droop
            VSC_filter = 0  # 0: RLC filter, 1: RL filter
            inner_control = 2  # 0:Direct modulation, 1: Voltage control, 2: Voltage and current control

            # current controller (kp and ki are calculated in the PSCAD model according to the parameters defined here)
            tau_c = 0.001  # time constant

            # Current saturation
            Imax = 1.1 #pu
            en_current_saturation = 1 # 1 = saturation enabled
                                      # 0 = saturation disabled

            # Voltage controller
            set_time_v = 0.05 # in s
            xi_v = 0.707 #damping
            wn_v = 4/(set_time_v*xi_v) # natural frequency
            kp_v = 2*xi_v*wn_v*filter_cap*1e-6*100
            ki_v = wn_v**2*filter_cap*1e-6
            print("kp_v =" + str(kp_v))
            print("ki_v =" + str(ki_v))    
            
            #kp_v = 4.789 # from optimisation
            #ki_v = 42.05
            
            # Active power regulation (in case that PQ-Droop is enabled)
            Qmax = 0.3
            mq_droop = 0.02/Qmax
            wf_droop = 50
            # in case that droop mode is activated for active power
            mp_droop = 0.05

            # Initial values
            Pvsc0 = genp[k];                    # Active power from power flow in MW
            Qvsc0 = genq[k];                    # Reactive power from power flow in Mvar
            Vvsc0 = voltage[k]*base_ACvoltage;  # Voltage from power flow in kV
            e_theta_vsc0 = angle[k]*math.pi/180-math.pi/2; # Bus angle from power flow in  rad

            Svsc0 = complex(Pvsc0, Qvsc0)   # Define S = P + jQ
            Vpoc0 = Vvsc0/math.sqrt(3)      # RMS ph-n voltage at point of connection (POC) 
            Itr = Svsc0/(3*Vpoc0)           # Current flowing through converter transformer
            Itr = Itr.conjugate()
            Vcap = Vpoc0 + Itr*complex(trafo_res, base_omega*trafo_induct) # Calculate the voltage at the filter capacitor (voltage control)
            Sref = 3*Vcap*Itr.conjugate()    # Power reference for the converter
            Delta_theta = math.atan(Vcap.imag/Vcap.real) # Angle of capacitor voltage (respect to POC)
            theta0 = e_theta_vsc0 + Delta_theta          # Angle of capacitor voltage (respect to slack bus)
            Vref = abs(Vcap)*math.sqrt(3)/base_ACvoltage # Voltage reference for the converter (pu)
            Icap = Vcap/complex(filter_res_cap, -1/(base_omega*filter_cap/1e6)) # Current flowing through the filter capacitor
            Ic = Itr+Icap                   # Current flowing through converter L filter
            Vc = Vcap + Ic*complex(filter_res, base_omega*filter_induct) # Calculate the voltage applied by the converter (internal voltage)
            vc_q0 = Vc.real*math.sqrt(2)    # q-axis converter voltage (used to initialize the converter as V source)
            vc_d0 = -Vc.imag*math.sqrt(2)   # d-axis converter voltage (used to initialize the converter as V source)
            vc_qc0 = math.cos(Delta_theta)*vc_q0-math.sin(Delta_theta)*vc_d0 # Rotation from POC to capacitor bus to express the qd-frame in the converter reference
            vc_dc0 = math.sin(Delta_theta)*vc_q0+math.cos(Delta_theta)*vc_d0 


            print("theta0 =" + str(theta0))
            print("Vref =" + str(Vref))
            print("Svsc0 =" + str(Svsc0))
            print("Sref =" + str(Sref))
            print("e_theta_vsc0 =" + str(e_theta_vsc0))
            print("vc_q0 =" + str(vc_q0))
            print("vc_d0 =" + str(vc_d0))
            print("vc_qc0 =" + str(vc_qc0))
            print("vc_dc0 =" + str(vc_dc0))
            
            # ----------- Send the variables to each model in PSCAD -----------
            print("Sending parameters to " +"GFOR" + str(k))
            exec("GFOR" + str(k) + ".set_parameters(f_1="+str(freqbase)+", Ab="+ str(base_power)+", Vb_ac_1="+str(base_ACvoltage)+", Vb_ac_2="+str(base_ACvoltage)+", Vb_dc="+str(base_DCvoltage)+
                                                    ", R_cf="+str(filter_res)+", L_cf="+str(filter_induct)+", C_cf="+str(filter_cap)+", Rcap_cf="+str(filter_res_cap)+
                                                    ",Imax_rms =" +str(Imax*base_current)+", limit_act ="+str(en_current_saturation)+
                                                    ",PQ_case =" +str(PQ_control) +",Cf_disable =" +str(VSC_filter) +",VC_case =" +str(inner_control) +
                                                    ",kp_v_input =" +str(kp_v)+", ki_v_input ="+str(ki_v)+", tau_c_loop ="+str(tau_c)+
                                                    ",mp_droop =" +str(mp_droop) +", wf_droop =" +str(wf_droop)+", mq_droop =" +str(mq_droop) +
                                                    ",tP1=100, tP2=100, tP3=100, tP4=100, tP5=100"+
                                                    ",Pref0=" +str(Sref.real/base_power)+", Pref1=0, Pref2=0, Pref3=0, Pref4=0, Pref5=0"+
                                                    ",tQ1=100, tQ2=100, tQ3=100, tQ4=100"+
                                                    ",Qref0="+str(Sref.imag/base_power)+ ",Qref1=0, Qref2=0, Qref3=0, Qref4=0"+
                                                    ",v_ref_pu="+str(Vref)+",theta0="+str(theta0)+",vc_qc0="+str(vc_qc0)+",vc_dc0="+str(vc_dc0)+
                                                    ",Ppoc="+str(Pvsc0)+ ",Qpoc="+str(Qvsc0)+",Vpoc="+str(Vvsc0/base_ACvoltage)+")")                      
            
                                
                                    #I_saturation=param_vsc.I_saturation, tsat_enable=param_vsc.tsat_enable, overflow_sampling=param_vsc.overflow_sampling,
                                    #gt_q=param_vsc.gt_q, gt_d=param_vsc.gt_d, f_saturation_up=param_vsc.f_saturation_up, f_saturation_lo=param_vsc.f_saturation_lo,
                                    #black_str_enable=param_vsc.black_str_enable)

        print("End of VSC-GFOR loop")
        # end of GFOL loop ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        # Run PSCAD
        project.run()
        
        print("Script is Done!")

    finally:
        #pscad.quit()
        pass
else:
    print("Failed to launch PSCAD")

# ------------------------------------------------------------------------------
#  End of script
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
