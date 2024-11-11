import os
import sys
import random
from datetime import datetime


from datagen.src.utils import save_dataframes, parse_setup_file, parse_args, concat_df_dict, save_results
from datagen.src.dimensions import Dimension
from datagen.src.objective_function_ACOPF import *
from datagen.src.sampling import gen_samples
from datagen.src.sampling import gen_cases
from datagen.src.sampling import eval_stability
from datagen.src.utils_obj_fun import *
from datagen.src.save_for_matlab import *

from stability_analysis.preprocess import preprocess_data, read_data, \
    process_raw
from stability_analysis.powerflow import GridCal_powerflow
from stability_analysis.powerflow.fill_d_grid_after_powerflow import fill_d_grid
from stability_analysis.preprocess.utils import *

try:
    from pycompss.api.task import task
    from pycompss.api.api import compss_wait_on
except ImportError:
    from datagen.dummies.task import task
    from datagen.dummies.api import compss_wait_on
    
from stability_analysis.operating_point_from_datagenerator import datagen_OP
from stability_analysis.modify_GridCal_grid import *
from datagen.src.save_for_matlab import save_full
#%%

Generators_NREL=pd.read_excel('Generators_NREL.xlsx')
Generators_NREL_onlyCIG=Generators_NREL.query('Pmax_CIG !=0 and Pmax_TOT_SG==0')

Generators=pd.read_excel('Generators_red.xlsx')
for i in range(len(Generators)):
    Generators.loc[i,'Num']=int(Generators.loc[i,'Node of connection'][4:])
    Generators.loc[i,'Type']=Generators.loc[i,'Generator Name'][:-3]
    bus_num=int(Generators.loc[i,'Node of connection'][4:])
    Generators.loc[i,'Generator Name'][:-3]
    # Generators.loc[i,'Region']=list(Buses.query('Num == @bus_num')['Region'])[0]

bus=104

Generators_bus=Generators.query('Num == @bus').rename(columns={'Generator Name':'Generator_Name'})
Generators_name=Generators_bus['Generator_Name']

#%%

from os import listdir
from os.path import isfile, join

def res_gen(res, GenNames, df):
    path='./input-files/Input files/RT/'+res+'/'
   
    files=[i.replace(' ','')+'RT.csv' for i in GenNames]
    
    for file in files:
        df_i=pd.read_csv(path+file)
        
        df_i["DATETIME"] = pd.to_datetime(df_i["DATETIME"])
        
        df_i=df_i.set_index(["DATETIME"]).sort_index()
        
        df[file[:-6]]=df_i
        
    return df
           
Solar=pd.DataFrame()
    
Solar=res_gen('Solar', Generators_name, Solar)
Solar['Day']=Solar.index.day
Solar['Month']=Solar.index.month
Solar['Hour']=Solar.index.hour

day=16
month=10

OPs=Solar.query('Day == @day and Month==@month')

#%%

path_data='../../stability_analysis/stability_analysis/data/'
Lines=pd.read_csv(path_data+'cases/IEEE_118_Lines.csv')
Lines_bus=Lines.query('Bus_from==@bus or Bus_to==@bus')

Line=Lines_bus.iloc[1]

#%%

Loads=pd.DataFrame()

for zone in [1,2,3]:

    Loads_R=pd.read_csv('./input-files/Input files/RT/Load/LoadR'+str(zone)+'RT.csv')
    
    Loads_R["DATETIME"] = pd.to_datetime(Loads_R["DATETIME"])
    
    Loads_R=Loads_R.set_index(["DATETIME"]).sort_index()
    
    Loads['R'+str(zone)]=Loads_R


Loads['TOT_load']=Loads.sum(axis=1)
Loads['Day']=Loads.index.day
Loads['Month']=Loads.index.month
Loads['Hour']=Loads.index.hour

OPs['Load_TOT']=Loads.query('Day == @day and Month==@month')['TOT_load']

#%%
excel_data = "IEEE_118_FULL"
raw = "IEEE118busNREL_TestDyn"
excel_headers = "Empty_template_V6"#"IEEE_118_FULL_headers"
excel_op = "OperationData_IEEE_118_NREL"

#%%

raw_file = os.path.join(path_data, "raw", raw + ".raw")
d_raw_data = process_raw.read_raw(raw_file)

# Preprocess input raw data to match Excel file format
preprocess_data.preprocess_raw(d_raw_data)

# %% Create GridCal Model
gridCal_grid = GridCal_powerflow.create_model(raw_file)

#%% Line
for line in gridCal_grid.lines:
    bf = int(line.bus_from.code)
    bt = int(line.bus_to.code)
    line.rate = Lines.loc[
        Lines.query('Bus_from == @bf and Bus_to == @bt').index[
            0], 'Max Flow (MW)']
    
 # %% READ EXCEL FILE
 
excel_sys = os.path.join(path_data, "cases", excel_headers + ".xlsx")
excel_sg = os.path.join(path_data, "cases", excel_data + "_data_sg.xlsx")
excel_vsc = os.path.join(path_data, "cases", excel_data + "_data_vsc.xlsx")
excel_op = os.path.join(path_data, "cases", excel_op + ".xlsx")

#%%
# Read data of grid elements from Excel file
d_grid, d_grid_0 = read_data.read_sys_data(excel_sys)
# TO BE DELETED
d_grid = read_data.tempTables(d_grid)

d_grid['T_trafo']=d_grid['T_trafo'].iloc[:0]


# %% READ EXEC FILES WITH SG AND VSC CONTROLLERS PARAMETERS
d_sg = read_data.read_data(excel_sg)
d_vsc = read_data.read_data(excel_vsc)

#%%

d_op = read_data.read_data(excel_op)
d_op['Buses']=d_op['Buses'].iloc[bus-2:bus].reset_index(drop=True)
d_op['Generators']=d_op['Generators'].query('BusNum == 103 or BusNum == 104').reset_index(drop=True)
d_op['Loads']=d_op['Loads'].query('Num==@bus').reset_index(drop=True)

OPs['Load']=float(d_op['Loads']['Load_Participation_Factor'])*OPs['Load_TOT']
OPs['TH']=OPs['Load']-OPs['Solar31']-OPs['Solar32']-OPs['Solar33']-OPs['Solar35']

#%% change SG values for Thevenin values
max_th=max(OPs['TH'])
min_th=min(OPs['TH'])

d_op['Generators'].loc[0,'Snom_SG']=1.5*max(max_th,abs(min_th))
d_op['Generators'].loc[0,'Snom']=d_op['Generators'].loc[0,'Snom_SG']+d_op['Generators'].loc[0,'Snom_CIG']
d_op['Generators'].loc[0,'Pmax']=1e6
d_op['Generators'].loc[0,'Pmin']=-1e6
d_op['Generators'].loc[0,'Qmax']=1e6
d_op['Generators'].loc[0,'Qmin']=-1e6
d_op['Generators'].loc[0,'Pmax_SG']=d_op['Generators'].loc[0,'Pmax']
d_op['Generators'].loc[0,'Pmin_SG']=d_op['Generators'].loc[0,'Pmin']


#%%

cases=pd.DataFrame()

#bus 103
cases['p_sg_Var0']=OPs['TH']
cases['q_sg_Var0']=OPs['TH']*0.33

cases['p_cig_Var0']=0
cases['q_cig_Var0']=0

cases['p_g_fol_Var0']=0
cases['p_g_for_Var0']=0

cases['q_g_fol_Var0']=0
cases['q_g_for_Var0']=0

#bus 104
cases['p_sg_Var1']=0
cases['q_sg_Var1']=0

cases['p_cig_Var1']=OPs['Solar31']+OPs['Solar33']+OPs['Solar32']+OPs['Solar35']
cases['q_cig_Var1']=cases['p_cig_Var0']*0.33

cases['p_load_Var0']=OPs['Load']
cases['q_load_Var0']=OPs['Load']*0.33

cases['p_g_fol_Var1']=np.array(OPs['Solar31'])+np.array(OPs['Solar32'])
#cases['p_gfol_Var1']=OPs['Solar32']

cases['p_g_for_Var1']=np.array(OPs['Solar33'])+np.array(OPs['Solar35'])
#cases['p_gfor_Var1']=OPs['Solar35']

cases['q_g_fol_Var1']=cases['p_g_fol_Var1']*0.33
# cases['q_gfol_Var1']=OPs['Solar32']*0.33
cases['q_g_for_Var1']=cases['p_g_for_Var1']*0.33
# cases['q_gfor_Var1']=OPs['Solar35']*0.33

#%%
case=cases.iloc[7]

#%%
d_raw_data, d_op = datagen_OP.generated_operating_point(case, d_raw_data,
                                                        d_op)
d_raw_data['generator']['Region']='R3'
#%%
slack_bus_num=103
d_raw_data['data_global'].loc[0,'ref_bus']=slack_bus_num
d_raw_data['data_global'].loc[0,'ref_element']='TH'

i_slack=int(d_raw_data['generator'].query('I == @slack_bus_num').index[0])

# slack_bus_num=80
assign_SlackBus_to_grid.assign_slack_bus(gridCal_grid, slack_bus_num)

idx_ind=np.zeros([2,2])
idx_ind[0,1]=103
idx_ind[1,0]=1
idx_ind[1,1]=104
assign_Generators_to_grid.assign_PVGen(GridCal_grid=gridCal_grid, d_raw_data=d_raw_data, d_op=d_op,voltage_profile_list=[1.05,0.9],indx_id=idx_ind)# V_set=1)

assign_PQ_Loads_to_grid.assign_PQ_load(gridCal_grid, d_raw_data)

#%%

# Get Power-Flow results with GridCal
pf_results = GridCal_powerflow.run_powerflow(gridCal_grid)

print('Converged:', pf_results.convergence_reports[0].converged_[0])


# Update PF results and operation point of generator elements
d_pf = process_powerflow.update_OP(gridCal_grid, pf_results, d_raw_data)
d_pf['info']=pd.DataFrame()
d_pf = additional_info_PF_results(d_pf, i_slack, pf_results, 1)

#%%


d_grid, d_opf = fill_d_grid_after_powerflow.fill_d_grid(d_grid,
                                                   gridCal_grid, d_pf,
                                                   d_raw_data, d_op)
    
d_grid['T_gen']['element']=d_grid['T_gen']['element'].replace('SG','TH')
d_grid['T_TH']=d_grid['T_gen'].query('element == "TH"').drop(['Rtr','Xtr'],axis=1)
d_grid['T_SG']=d_grid['T_SG'].iloc[:0]

folder_path = '../version6_original/version6/Tool/01_data/cases/TestDyn_2CIGs'
filename='OP_7'

#%% fix d_grid

d_grid['T_buses']['bus']=[1,2]
d_grid['T_buses']['SyncArea']=1

d_grid['T_gen'].loc[d_grid['T_gen'].query('bus == 103').index,'bus']=1
d_grid['T_gen'].loc[d_grid['T_gen'].query('bus == 104').index,'bus']=2

d_grid['T_gen'].loc[d_grid['T_gen'].query('element == "GFOL"').index,'Sn']=sum(Generators_bus.query('Generator_Name == "Solar 31" or Generator_Name == "Solar 32"')['Max Capacity (MW)'])/0.95
d_grid['T_gen'].loc[d_grid['T_gen'].query('element == "GFOR"').index,'Sn']=sum(Generators_bus.query('Generator_Name == "Solar 33" or Generator_Name == "Solar 35"')['Max Capacity (MW)'])/0.95
d_grid['T_gen']['SyncArea']=1

d_grid['T_global']['SyncArea']=1
d_grid['T_global'].loc[0,'ref_bus']=1

d_grid['T_NET']['Area']=1
d_grid['T_NET']['SyncArea']=1
d_grid['T_NET'].loc[d_grid['T_NET'].query('bus_from==103').index,'bus_from']=1
d_grid['T_NET'].loc[d_grid['T_NET'].query('bus_from==104').index,'bus_from']=2
d_grid['T_NET'].loc[d_grid['T_NET'].query('bus_to==103').index,'bus_to']=1
d_grid['T_NET'].loc[d_grid['T_NET'].query('bus_to==104').index,'bus_to']=2

d_grid['T_TH']['bus']=[1 if bus == 103 else 2 for bus in list(d_grid['T_TH']['bus'])]
d_grid['T_TH']['SyncArea']=1
d_grid['T_TH']['R']=0.01
d_grid['T_TH']['X']=0.1

d_grid['T_VSC']['bus']=[1 if bus == 103 else 2 for bus in list(d_grid['T_VSC']['bus'])]
d_grid['T_VSC']['SyncArea']=1

for n in list(d_grid['T_VSC']['number']):
    d_grid['T_VSC'].loc[d_grid['T_VSC'].query('number == @n').index,'Sn']=float(d_grid['T_gen'].query('number == @n and element != "TH"')['Sn'])
        
d_pf['pf_bus']['bus']=[1,2]
d_pf['pf_bus']['Area']=1
d_pf['pf_bus']['SyncArea']=1
    
d_grid['T_load']['bus']=[1 if bus == 103 else 2 for bus in list(d_grid['T_load']['bus'])]
d_grid['T_load']['SyncArea']=1
d_grid['T_load']['Area']=1


#%%

save_full(folder_path,filename,d_pf,d_grid)

#%%
import copy

d_grid_4CIGs=copy.copy(d_grid)
cosphi=d_pf['pf_gen']['cosphi'].iloc[1]

d_grid_4CIGs['T_gen']=pd.concat([d_grid_4CIGs['T_gen'],d_grid_4CIGs['T_gen'].query('element == "GFOL"')],axis=0)
d_grid_4CIGs['T_gen']=pd.concat([d_grid_4CIGs['T_gen'],d_grid_4CIGs['T_gen'].query('element == "GFOR"')],axis=0)
d_grid_4CIGs['T_gen']=d_grid_4CIGs['T_gen'].sort_values(by='element').reset_index(drop=True)
d_grid_4CIGs['T_gen'].query('element != "TH"')['number']=np.arange(1,len(d_grid_4CIGs['T_gen']))

gfol=d_grid_4CIGs['T_gen'].query('element == "GFOL"')

def change_T_gen(T_gen,table_rows,Generators_name,Generators_bus,OPs,cosphi):
    gen_count=1
    for i in table_rows.index:
        gen_name=Generators_name.iloc[i]
        table_rows.loc[i,'Sn']=float(Generators_bus.query('Generator_Name==@gen_name')['Max Capacity (MW)'])/0.95
        gen_name=gen_name.replace(' ','')
        table_rows.loc[i,'P']=OPs.iloc[7][gen_name]/100
        table_rows.loc[i,'Q']=OPs.iloc[7][gen_name]*np.tan(np.arccos(cosphi))/100
        T_gen.loc[i]=table_rows.loc[i]
        T_gen.loc[i,'number']=gen_count
        gen_count=gen_count+1
        
    return T_gen#, gen_count

# gen_count=0
d_grid_4CIGs['T_gen']=change_T_gen(d_grid_4CIGs['T_gen'],gfol,Generators_name,Generators_bus,OPs,cosphi)

gfor=d_grid_4CIGs['T_gen'].query('element == "GFOR"')
d_grid_4CIGs['T_gen']=change_T_gen(d_grid_4CIGs['T_gen'],gfor,Generators_name,Generators_bus,OPs,cosphi)

d_grid_4CIGs['T_VSC']=d_grid_4CIGs['T_VSC'].iloc[:0]
d_grid_4CIGs['T_VSC']=pd.concat([d_grid_4CIGs['T_VSC'],d_grid_4CIGs['T_gen'].query('element!="TH"')],axis=0)

for i in range(0,len(d_grid_4CIGs['T_VSC'])):
    area=d_grid_4CIGs['T_VSC'].loc[i,'Area']
    d_grid_4CIGs['T_VSC'].loc[i,'SyncArea']=area
    d_grid_4CIGs['T_VSC'].loc[i,'Vn']=float(d_grid_4CIGs['T_global'].query('Area==@area')['Vb_kV'])
    
d_grid_4CIGs['T_VSC']=d_grid_4CIGs['T_VSC'].drop(['mode'],axis=1)
d_grid_4CIGs['T_VSC']=d_grid_4CIGs['T_VSC'].rename(columns={'element':'mode'})

d_grid_4CIGs['T_load']['Area']=1
d_grid_4CIGs['T_load']['SyncArea']=1

folder_path = '../version6_original/version6/Tool/01_data/cases/TestDyn_4CIGs'
filename='OP_7'
save_full(folder_path,filename,d_pf,d_grid_4CIGs)
