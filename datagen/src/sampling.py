#  Copyright 2002-2023 Barcelona Supercomputing Center (www.bsc.es)

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""This module serves as a comprehensive data generator based on the entropy
of different regions of space. The main objectives are to produce samples
and cases from a given set of dimensions and then evaluate these cases to
determine their stability. Parallel execution is used to evaluate the
stability of each case.
"""

import numpy
import numpy as np

try:
    from pycompss.api.task import task
    from pycompss.api.api import compss_wait_on
    from pycompss.api.constraint import constraint
except ImportError:
    from datagen.dummies.task import task
    from datagen.dummies.api import compss_wait_on
    from datagen.dummies.constraint import constraint



def generate_columns(dim):
    """Assigns names for every variable in a dimension.

    :param dim: Involved dimension
    :return: Names of de variable_borders
    """
    if (isinstance(dim.variable_borders, numpy.ndarray) and
            not np.all(np.isnan(dim.variable_borders))):
        return [f"{dim.label}_Var{v}" for v in
                range(len(dim.variable_borders))]
    else:
        if dim.values:
            return [f"{dim.label}_Var{v}" for v in range(len(dim.values))]
        else:
            return None


def gen_voltage_profile(vmin,vmax,delta_v,d_raw_data,slack_bus,GridCal_grid,generator):
    from GridCalEngine.DataStructures.numerical_circuit import \
        compile_numerical_circuit_at

    # Find Index of slack bus
    i_slack_bus= d_raw_data['results_bus'].query('I==@slack_bus').index[0]

    # Assign random voltage to slack bus
    V = generator.random() * (vmax - vmin) + vmin
    # d_raw_data['generator'].loc[d_raw_data['generator'].query('I == @slack_bus').index[0],'V']=V

    # Initialize voltage matrix
    # adj_matrix=GridCal_grid.get_adjacent_matrix()
    main_circuit=compile_numerical_circuit_at(GridCal_grid)
    C=main_circuit.Cf+main_circuit.Ct
    adj_matrix=C.T.dot(C)


    # Find buses adjacents to slack_bus
    idx_adj_buses=list(GridCal_grid.get_adjacent_buses(adj_matrix, int(i_slack_bus)))
    adj_buses= d_raw_data['results_bus'].loc[idx_adj_buses,'I']

    # Relation table between index and ID for all the nodes
    indx_id=np.zeros([len(d_raw_data['results_bus']),2])
    indx_id[:,0]=d_raw_data['results_bus'].index
    indx_id[:,1]=d_raw_data['results_bus']['I']


    # initialize generators_id and generators_index
    generators_id = d_raw_data['generator']['I'].tolist()
    generators_index_list = []
    for row in indx_id:
        if row[1] in generators_id:
            generators_index_list.append(row[0])


    # Choose start parameters
    start_node = i_slack_bus
    pending = [start_node]
    # node: distancia, num_modificacions 
    distances = {start_node:[0,1]}
    # Initialize a row to store de voltages calculated, with a length equal to the number of nodes
    voltages = [0]*len(indx_id)
    voltages[start_node]=V

    while len(pending)>0:
        to_calculate = []
        current_node = pending[0]
        current_node_distance = distances[current_node][0]
        adjacent_nodes_list = list(GridCal_grid.get_adjacent_buses(adj_matrix, int(current_node)))
        for adjacent_node in adjacent_nodes_list:
            if adjacent_node != current_node:
        # for adjacent_node, num in enumerate(adj_matrix[current_node]):
            # if num == 1:
                adjacent_node_distance = distances.get(adjacent_node, None)
                if adjacent_node_distance is None :
                    to_calculate.append(adjacent_node)
                    pending.append(adjacent_node)
                    distances[adjacent_node]=[current_node_distance+1,0]
                elif adjacent_node_distance[0] > current_node_distance:
                    to_calculate.append(adjacent_node)
        for adjacent_node in to_calculate:
            calculate_voltage(adjacent_node, current_node, delta_v, distances, generators_index_list, voltages, vmin, vmax, generator)
    #    print(f"El node {current_node} donara a {to_calculate}")
        current_bus=d_raw_data['results_bus'].loc[current_node,'I']
        to_calculate_bus=list(d_raw_data['results_bus'].loc[to_calculate,'I'])
        # print(f"El bus {current_bus} donara a {to_calculate_bus}")
        pending.pop(0)
        # print(voltages)
    # print(distances)

    return voltages, indx_id

def calculate_voltage(adjacent_node, current_node, delta_v, distances, generators_index_list, voltages, vmin, vmax, generator):
    # Aumento en una unitat el nombre de modificacions
    distances[adjacent_node][1]+=1
    num_modifications = distances[adjacent_node][1]
    # Miro si el node contigu es generador
    if adjacent_node in generators_index_list:
        V = generator.random() * (delta_v + delta_v) -delta_v + voltages[current_node]
        if num_modifications == 1:
            V_adjacent_node = V
        elif num_modifications > 1:
            V_adjacent_node = (num_modifications-1)*(voltages[adjacent_node]/num_modifications) + V/num_modifications
        if V_adjacent_node < vmin:
            voltages[adjacent_node]=vmin
        elif V_adjacent_node > vmax:
            voltages[adjacent_node]=vmax
        else:
            voltages[adjacent_node]=V_adjacent_node
    else:
        voltages[adjacent_node]=voltages[current_node]

