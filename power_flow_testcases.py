import GridCalEngine as gce
from datagen.src.save_for_matlab import save_full, modify_full
from stability_analysis.preprocess import preprocess_data, read_data, \
    process_raw
from stability_analysis.powerflow import GridCal_powerflow
from stability_analysis.powerflow.fill_d_grid_after_powerflow import fill_d_grid
from stability_analysis.preprocess.utils import *
from stability_analysis.powerflow import *
#%%

# declare a circuit object
grid = gce.MultiCircuit()

# Add the buses and the generators and loads attached
bus1 = gce.Bus('Bus 1', vnom=138, code='1')
bus1.is_slack = True  # we may mark the bus a slack
grid.add_bus(bus1)

# add a generator to the bus 1
gen1 = gce.Generator('Slack Thevenin', vset=1.0, Snom=100)
grid.add_generator(bus1, gen1)

# add bus 2 with a load attached
bus2 = gce.Bus('Bus 2', vnom=138, code='2')
grid.add_bus(bus2)
gen2 = gce.Generator('3 CIGs', vset=1.01, Snom=100, P=35)
grid.add_generator(bus2, gen2)

# # add bus 3 with a load attached
# bus3 = gce.Bus('Bus 3', Vnom=20)
# grid.add_bus(bus3)
# grid.add_load(bus3, gce.Load('load 3', P=25, Q=15))

# # add bus 4 with a load attached
# bus4 = gce.Bus('Bus 4', Vnom=20)
# grid.add_bus(bus4)
# grid.add_load(bus4, gce.Load('load 4', P=40, Q=20))

# # add bus 5 with a load attached
# bus5 = gce.Bus('Bus 5', Vnom=20)
# grid.add_bus(bus5)
# grid.add_load(bus5, gce.Load('load 5', P=50, Q=20))

# add Lines connecting the buses
grid.add_line(gce.Line(bus1, bus2, name='line 1-2', r=0.0466, x=0.1584, b=0.0407))
# grid.add_line(gce.Line(bus1, bus3, name='line 1-3', r=0.05, x=0.11, b=0.02))
# grid.add_line(gce.Line(bus1, bus5, name='line 1-5', r=0.03, x=0.08, b=0.02))
# grid.add_line(gce.Line(bus2, bus3, name='line 2-3', r=0.04, x=0.09, b=0.02))
# grid.add_line(gce.Line(bus2, bus5, name='line 2-5', r=0.04, x=0.09, b=0.02))
# grid.add_line(gce.Line(bus3, bus4, name='line 3-4', r=0.06, x=0.13, b=0.03))
# grid.add_line(gce.Line(bus4, bus5, name='line 4-5', r=0.04, x=0.09, b=0.02))

#%%

# Get Power-Flow results with GridCal
pf_results = GridCal_powerflow.run_powerflow(grid)

print('Converged:', pf_results.convergence_reports[0].converged_[0])

#%%

pf_bus, pf_load, pf_gen = process_powerflow.process_GridCal_PF_loadPQ(grid, pf_results)
pf_bus['Area']=1
pf_bus['SyncArea']=1

d_pf = {'pf_bus':pf_bus, 'pf_load': pf_load, 'pf_gen': pf_gen}

#%%
np.tan(np.arccos(0.989999))
#maually put the results of the PF




#%%
# path_data='C:/Users/Francesca/miniconda3/envs/hp2c-dt/datagen_testdyn/version6_original/version6/Tool/01_data/cases/TestDyn_example_model/'

# excel_data = "scenario_0"
# excel_headers = "scenario_0"#"IEEE_118_FULL_headers"

#  # %% READ EXCEL FILE

# import os
 
# excel_sys = os.path.join(path_data, "", excel_headers + ".xlsx")
# excel_sg = os.path.join(path_data, "", excel_data + "_data_sg.xlsx")
# excel_vsc = os.path.join(path_data, "", excel_data + "_data_vsc.xlsx")

# #%%
# # Read data of grid elements from Excel file
# # %%
# d_grid, d_grid_0 = read_data.read_sys_data(excel_sys)

# d_grid['T_trafo']=d_grid['T_trafo'].iloc[:0]


# # %% READ EXEC FILES WITH SG AND VSC CONTROLLERS PARAMETERS
# d_sg = read_data.read_data(excel_sg)
# d_vsc = read_data.read_data(excel_vsc)

#%%
