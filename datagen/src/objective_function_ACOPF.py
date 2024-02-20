import math
import time
import numpy as np
import copy

from datagen.src import sampling

from .utils import get_case_results
from stability_analysis.operating_point_from_datagenerator import datagen_OP
from stability_analysis.modify_GridCal_grid import assign_Generators_to_grid,assign_PQ_Loads_to_grid, assign_SlackBus_to_grid
from stability_analysis.powerflow import GridCal_powerflow, process_powerflow, slack_bus, fill_d_grid_after_powerflow
from stability_analysis.preprocess import preprocess_data, read_data, process_raw, parameters,read_op_data_excel, admittance_matrix
from stability_analysis.state_space import generate_NET, build_ss, generate_elements
from stability_analysis.analysis import small_signal

from stability_analysis.powerflow import check_feasibility

from GridCalEngine.Simulations.PowerFlow.power_flow_options import ReactivePowerControlMode, SolverType
from GridCalEngine.Simulations.OPF.NumericalMethods import ac_opf
from GridCalEngine.Simulations.OPF.NumericalMethods.ac_opf import run_nonlinear_opf, ac_optimal_power_flow
from GridCalEngine.Core.DataStructures.numerical_circuit import compile_numerical_circuit_at
import GridCalEngine.api as gce

from .utils_obj_fun import *

def feasible_power_flow_ACOPF(case, **kwargs):
    d_raw_data = kwargs.get("d_raw_data", None)
    d_op = kwargs.get("d_op", None)
    GridCal_grid = kwargs.get("GridCal_grid", None)
    d_grid = kwargs.get("d_grid", None)
    d_sg = kwargs.get("d_sg", None)
    d_vsc = kwargs.get("d_vsc", None)
    voltage_profile = kwargs.get("voltage_profile", None)
    v_min_v_max_delta_v = kwargs.get("v_min_v_max_delta_v", None)
    V_set = kwargs.get("V_set", None)

    if voltage_profile != None and v_min_v_max_delta_v == None:
        print('Error: Voltage profile option selected but v_min, v_max, and delta_v are missing')
        return None, None, None

    if voltage_profile != None and V_set != None:
        print('Error: Both Voltage profile and V_set option is selected. Choose only one of them')
        return None, None, None

    if voltage_profile == None and V_set == None:
        print('Error: Neither Voltage profile or V_set option is selected. Choose one of them')
        return None, None, None

    d_raw_data, d_op = datagen_OP.generated_operating_point(case, d_raw_data,
                                                            d_op)
    d_raw_data, slack_bus_num = choose_slack_bus(d_raw_data)
    # slack_bus_num=80
    assign_SlackBus_to_grid.assign_slack_bus(GridCal_grid, slack_bus_num)

    if voltage_profile != None:
        vmin = v_min_v_max_delta_v[0]
        vmax = v_min_v_max_delta_v[1]
        delta_v = v_min_v_max_delta_v[2]

        voltage_profile_list, indx_id = sampling.gen_voltage_profile(vmin, vmax, delta_v, d_raw_data, slack_bus_num,
                                                                     GridCal_grid)

        assign_Generators_to_grid.assign_PVGen(GridCal_grid=GridCal_grid, d_raw_data=d_raw_data, d_op=d_op,
                                               voltage_profile_list=voltage_profile_list, indx_id=indx_id)

    elif V_set != None:
        assign_Generators_to_grid.assign_PVGen(GridCal_grid=GridCal_grid, d_raw_data=d_raw_data, d_op=d_op, V_set=V_set)

    assign_PQ_Loads_to_grid.assign_PQ_load(GridCal_grid, d_raw_data)

    nc = compile_numerical_circuit_at(GridCal_grid)
    nc.generator_data.cost_0[:] = 0
    nc.generator_data.cost_1[:] = 0
    nc.generator_data.cost_2[:] = 0
    pf_options = gce.PowerFlowOptions(solver_type=gce.SolverType.NR, verbose=1, tolerance=1e-8)
    ac_optimal_power_flow(nc=nc, pf_options=pf_options, plot_error=True)

    GridCal_grid.get_bus_branch_connectivity_matrix()
    nc = compile_numerical_circuit_at(GridCal_grid)
    print('')

    # return d_pf_original, d_pf, d_raw_data