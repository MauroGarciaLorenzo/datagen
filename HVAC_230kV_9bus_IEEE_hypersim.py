import os
import yaml

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import GridCalEngine.Devices as gc
from GridCalEngine.Devices.multi_circuit import MultiCircuit
from stability_analysis.powerflow import GridCal_powerflow

lines_info = pd.read_excel('./hypersim9buses/Lineas_9bus_IEEE_hypersim.xlsx')
trafos_info = pd.read_excel('./hypersim9buses/Trafos_9bus_IEEE_hypersim.xlsx')
#%%
gridCal_grid = MultiCircuit()
v_b = 230
S_b=100
f_b=60
gridCal_grid.Sbase=S_b
gridCal_grid.fBase=f_b

bus1 = gc.Bus('Bus 1', vnom=24)
bus1.code=1
bus1.is_slack = True
gridCal_grid.add_bus(bus1)

bus2 = gc.Bus('Bus 2', vnom=18) 
bus2.code=2
gridCal_grid.add_bus(bus2)

bus3 = gc.Bus('Bus 3', vnom=15.5)
bus3.code=3
gridCal_grid.add_bus(bus3)

bus4 = gc.Bus('Bus 4', vnom=v_b)
bus4.code=4
gridCal_grid.add_bus(bus4)

bus5 = gc.Bus('Bus 5', vnom=v_b)
bus5.code=5
gridCal_grid.add_bus(bus5)

bus6 = gc.Bus('Bus 6', vnom=v_b)
bus6.code=6
gridCal_grid.add_bus(bus6)

bus7 = gc.Bus('Bus 7', vnom=v_b)
bus7.code=7
gridCal_grid.add_bus(bus7)

bus8 = gc.Bus('Bus 8', vnom=v_b)
bus8.code=8
gridCal_grid.add_bus(bus8)

bus9 = gc.Bus('Bus 9', vnom=v_b)
bus9.code=9
gridCal_grid.add_bus(bus9)

for bus in gridCal_grid.get_buses():
    bus.Vm0=1
    bus.Va0=0
    # if bus.code=='1':
    #     bus.Vm0=1.04
    #     bus.Va0=0
    # elif bus.code=='2' or bus.code=='3' or bus.code=='4' or bus.code=='7' or bus.code=='9':
    #     bus.Vm0=1.03#2579
    #     bus.Va0=0
    # elif bus.code=='5':
    #     bus.Vm0=1#0.99563
    #     bus.Va0=0
    # elif bus.code=='6':
    #     bus.Vm0=1.01#265
    #     bus.Va0=0
    # elif bus.code=='8':
    #     bus.Vm0= 1.02#15884            
    #     bus.Va0=0
    # # elif bus.code=='9':
    # #     bus.Vm0=1.03235
    # #     bus.Va0=0

gen1 = gc.Generator('Slack Generator')
gridCal_grid.add_generator(bus1, gen1)

gen2 = gc.Generator('Generator 2')
gridCal_grid.add_generator(bus2, gen2)

gen3 = gc.Generator('Generator 3')
gridCal_grid.add_generator(bus3, gen3)
    
Snom_gen=[512,270,125]
Pgen=[72,163,85]
V_set=[1.04,1.025,1.025]

for idx,gen in enumerate(gridCal_grid.get_generators()):
    gen.Pmax=1e3
    gen.Qmax=1e3
    gen.Qmin=-1e3
    
    gen.P=Pgen[idx]
    gen.Snom=Snom_gen[idx]
    gen.Vset=V_set[idx]

gridCal_grid.add_load(bus5, gc.Load('load 5', P=125, Q=50))
gridCal_grid.add_load(bus6, gc.Load('load 6', P=90, Q=30))
gridCal_grid.add_load(bus8, gc.Load('load 8', P=100, Q=35))

Pg_tot=sum([gen.P for gen in gridCal_grid.get_generators()]) 
Pl_tot=sum([load.P for load in gridCal_grid.get_loads()]) 
Ql_tot=sum([load.Q for load in gridCal_grid.get_loads()]) 

gridCal_grid.add_line(gc.Line(bus4, bus5, 'line 4-5', r=lines_info.loc[0,'Rpu'], x=lines_info.loc[0,'Xpu'], b=lines_info.loc[0,'Bpu'], rate=1e6))
gridCal_grid.add_line(gc.Line(bus4, bus6, 'line 4-6', r=lines_info.loc[1,'Rpu'], x=lines_info.loc[1,'Xpu'], b=lines_info.loc[1,'Bpu'], rate=1e6))
gridCal_grid.add_line(gc.Line(bus5, bus7, 'line 5-7', r=lines_info.loc[2,'Rpu'], x=lines_info.loc[2,'Xpu'], b=lines_info.loc[2,'Bpu'], rate=1e6))
gridCal_grid.add_line(gc.Line(bus6, bus9, 'line 6-9', r=lines_info.loc[3,'Rpu'], x=lines_info.loc[3,'Xpu'], b=lines_info.loc[3,'Bpu'], rate=1e6))
gridCal_grid.add_line(gc.Line(bus7, bus8, 'line  7-8', r=lines_info.loc[4,'Rpu'], x=lines_info.loc[4,'Xpu'], b=lines_info.loc[4,'Bpu'], rate=1e6))
gridCal_grid.add_line(gc.Line(bus8, bus9, 'line 8-9', r=lines_info.loc[5,'Rpu'], x=lines_info.loc[5,'Xpu'], b=lines_info.loc[5,'Bpu'], rate=1e6))

gridCal_grid.add_transformer2w(gc.Transformer2W(bus1,bus4))    
gridCal_grid.add_transformer2w(gc.Transformer2W(bus2,bus7))    
gridCal_grid.add_transformer2w(gc.Transformer2W(bus3,bus9))    

trafo_x=[0.0576,0.0625,0.0586]
for tt,trafo in enumerate(gridCal_grid.transformers2w):
    bf = int(trafo.bus_from.code)
    bt = int(trafo.bus_to.code)
    #t_info=trafos_info.query("bus_from == @bf and bus_to==@bt")
    idx_trafo=trafos_info.query("bus_from == @bf and bus_to==@bt").index[0]
   
    w=1#2*np.pi*50
    trafos_info.loc[idx_trafo, 'Zp']=complex(trafos_info.loc[idx_trafo, 'Rp_pu'],trafos_info.loc[idx_trafo, 'Lp_pu']*w)
    trafos_info.loc[idx_trafo, 'Zs']=complex(trafos_info.loc[idx_trafo, 'Rs_pu'],trafos_info.loc[idx_trafo, 'Ls_pu']*w)
    trafos_info.loc[idx_trafo, 'Zs"']=trafos_info.loc[idx_trafo, 'Zs']*(trafos_info.loc[idx_trafo, 'Vp']/trafos_info.loc[idx_trafo, 'Vs'])**2
    #trafos_info.loc[idx_trafo, 'Zm']=1/(1/trafos_info.loc[idx_trafo, 'Rm']+1/complex(0,2*np.pi*trafos_info.loc[idx_trafo, 'Lm']))
    trafos_info.loc[idx_trafo, 'Zm_par']=complex(0,trafos_info.loc[idx_trafo, 'Rm_pu']*trafos_info.loc[idx_trafo, 'Lm_pu']*w)/complex(trafos_info.loc[idx_trafo, 'Rm_pu'],trafos_info.loc[idx_trafo, 'Lm_pu']*w)
    
    Z1= trafos_info.loc[idx_trafo, 'Zs"']+trafos_info.loc[idx_trafo, 'Zp']
    Z2= trafos_info.loc[idx_trafo, 'Zm_par']
    
    # Z3=Z1+1/Z2
    # trafos_info.loc[idx_trafo, 'R']=np.real(Z3)#np.real(trafos_info.loc[idx_trafo, 'Zp']+trafos_info.loc[idx_trafo, 'Zs"']+trafos_info.loc[idx_trafo, 'Zm_par'])
    # trafos_info.loc[idx_trafo, 'X']=np.imag(Z3)#np.imag(trafos_info.loc[idx_trafo, 'Zp']+trafos_info.loc[idx_trafo, 'Zs"']+trafos_info.loc[idx_trafo, 'Zm_par'])
    
    Zbase=1#trafos_info.loc[idx_trafo, 'Vp']**2/gridCal_grid.Sbase/1e6
    trafos_info.loc[idx_trafo, 'R_pu']=np.real(Z1)/Zbase
    trafos_info.loc[idx_trafo, 'X_pu']=np.imag(Z1)/Zbase
    trafos_info.loc[idx_trafo, 'G_pu']=1/trafos_info.loc[idx_trafo, 'Rm_pu']#np.real(1/Z2)*Zbase
    trafos_info.loc[idx_trafo, 'B_pu']=1/trafos_info.loc[idx_trafo, 'Lm_pu']#abs(np.imag(1/Z2)*Zbase)
    
    trafo.R=0#np.real(Z1)/Zbase
    trafo.X=trafo_x[tt]#np.imag(Z1)/Zbase
    
    trafo.G=0#np.real(1/Z2)*Zbase
    trafo.B=0#np.imag(1/Z2)*Zbase
    
    # trafo.R=trafos_info.loc[idx_trafo, 'R_pu']
    # trafo.X=trafos_info.loc[idx_trafo, 'X_pu']
    
    trafo.rate = 1e10
    
    
    # trafo.rate = lines_ratings.loc[
    #     lines_ratings.query('Bus_from == @bf and Bus_to == @bt').index[
    #         0], 'Max Flow (MW)']

    
#%%
from GridCalEngine.Simulations.PowerFlow.power_flow_options import ReactivePowerControlMode, SolverType

    
# Get Power-Flow results with GridCal
#SolverType.IWAMOTO, SolverType.NR, SolverType.LM, SolverType.FASTDECOUPLED
pf_results = GridCal_powerflow.run_powerflow(gridCal_grid,solver_type=SolverType.NR, Qconrol_mode=ReactivePowerControlMode.NoControl)

print('Converged:', pf_results.convergence_reports[0].converged_[0])

from GridCalEngine.Simulations.PowerFlow.power_flow_results import PowerFlowResults
from stability_analysis.powerflow.process_powerflow import process_GridCal_PF_loadPQ
pf_bus, pf_load, pf_gen = process_GridCal_PF_loadPQ(gridCal_grid, pf_results)
