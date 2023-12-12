from GridCalEngine.IO.file_handler import FileOpen
from GridCalEngine.Simulations.PowerFlow.power_flow_worker import PowerFlowOptions
from GridCalEngine.Simulations.PowerFlow.power_flow_options import ReactivePowerControlMode, SolverType
from GridCalEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver

import numpy as np
import pandas as pd
from os import path


def create_model(path_raw,name_raw):
    
    # GET GRID TOPOLOGY
    raw_file = path.join(path_raw, name_raw)  
    grid = FileOpen(raw_file).open()
    return grid
    
def run_powerflow(grid):
            
    # RUN POWERFLOW
    for solver_type in [SolverType.NR]: #, SolverType.IWAMOTO, SolverType.LM, SolverType.FASTDECOUPLED]:
    
        print(solver_type)
    
        options = PowerFlowOptions(solver_type,
                                   verbose=False,
                                   initialize_with_existing_solution=False,
                                   retry_with_other_methods=False,
                                   ignore_single_node_islands = True,
                                   tolerance = 1e-10)
    
    pf = PowerFlowDriver(grid, options)
    pf.run()
    
    grid.get_buses()
    return pf

