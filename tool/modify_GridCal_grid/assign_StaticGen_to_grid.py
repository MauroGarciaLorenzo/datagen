import numpy as np
import pandas as pd
import numpy as np
import GridCalEngine.api as gce


def assign_StaticGen(GridCal_grid, d_raw_data, d_op):
    
    for bus in GridCal_grid.get_buses():
        if bus.is_slack:
            bus_code=int(bus.code)
            gen=bus.generators[0]
            
            gen.Snom=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Snom']
            gen.P=d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @bus_code').index[0],'PG']
            gen.Qmax=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Qmax']
            gen.Qmin=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Qmin']
            gen.Pmax=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Pmax']
            gen.Pmin=d_op['Generators'].loc[d_op['Generators'].query('BusNum == @bus_code').index[0],'Pmin']
            gen.Vset=1
            
        elif len(bus.generators)!=0:
            
            bus_code=int(bus.code)
            gen_name=bus.generators[0].name
                        
            GridCal_grid.add_static_generator(bus, gce.StaticGenerator(gen_name, P=d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @bus_code').index[0],'PG'], Q=d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @bus_code').index[0],'QG']))
            
            bus.generators=[]
            
    
