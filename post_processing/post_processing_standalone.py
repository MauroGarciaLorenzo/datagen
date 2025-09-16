import os
from utils_pp_standalone import *
import matplotlib.pyplot as plt

#ncig_list=['10','11','12']
path='../results/MareNostrum'

#dir_names=[dir_name for dir_name in os.listdir(path) if dir_name.startswith('datagen') and 'zip' not in dir_name]#

dir_names=['datagen_ACOPF_slurm23172357_cu10_nodes32_LF09_seed3_nc3_ns500_d7_20250627_214226_7664-20250630T085420Z-1-005']

for dir_name in dir_names:
    path_results = os.path.join(path, dir_name)
    
    results_dataframes, csv_files= open_csv(path_results)
    
    perc_stability(results_dataframes['case_df_op'],dir_name)


#%% ---- VOLTAGE PROFILE ANALYSIS ----
df_op_conv=results_dataframes['case_df_op'].query('Stability >=0')
voltage_conv=df_op_conv[[c for c in df_op_conv.columns if c.startswith('V')]]

fig=plt.figure()
ax=fig.add_subplot()
for ind in voltage_conv.index:
    ax.plot(voltage_conv.loc[ind])    
    
df_op_stab=results_dataframes['case_df_op'].query('Stability ==1')
voltage_stab=df_op_stab[[c for c in df_op_stab.columns if c.startswith('V')]]

df_op_unstab=results_dataframes['case_df_op'].query('Stability ==0')
voltage_unstab=df_op_unstab[[c for c in df_op_unstab.columns if c.startswith('V')]]

fig=plt.figure()
ax=fig.add_subplot()
for ind in voltage_unstab.index:
    ax.plot(voltage_unstab.loc[ind], '--r')    
for ind in voltage_stab.index:
    ax.plot(voltage_stab.loc[ind], '--g')    

ax.set_xticks(np.arange(0,118,10))


#%%
import pandapower.networks as pn
import pandapower.plotting as pp

# 2. Load the built‑in IEEE‑118‑bus grid
net = pn.case118()          # returns a fully populated pandapower net

# 3. Generate synthetic coordinates (the MATPOWER case has none)
pp.create_generic_coordinates(net, overwrite=True)

# 4. Static Matplotlib sketch – fastest way to eyeball the topology
pp.simple_plot(net, plot_loads=True)

# 5. Optional: interactive Plotly version (pan/zoom, hover tooltips)
from pandapower.plotting.plotly import simple_plotly
simple_plotly(net, use_bus_geodata=True)

#%%
import pandas as pd
import pandapower.networks as pn
import pandapower.plotting as pp

# Load the grid
net = pn.case118()

# Load real bus coordinates
coords_df = pd.read_csv("Bus118.csv", index_col=0)  # adjust path as needed

# Assign coordinates to buses
bus_geodata = {bus: (coords_df.loc[bus, "x"], coords_df.loc[bus, "y"]) for bus in coords_df.index}
net.bus_geodata = bus_geodata

# Plot
pp.simple_plot(net, plot_loads=True)

#%%
import numpy as np
import pandapower.networks as pn
import pandapower.plotting as pp

# ------------------------------------------------------------------
# 1. load the grid
# ------------------------------------------------------------------
net = pn.case118()

# ------------------------------------------------------------------
# 2. build a schematic coordinate map ("typical" 4‑block layout)
# ------------------------------------------------------------------
# centre‑points for the four blocks (x, y)
block_centres = {1: (-60,  60),   # NW
                 2: ( 60,  60),   # NE
                 3: (-60, -60),   # SW
                 4: ( 60, -60)}   # SE

# how tightly to cluster buses around the block centre
radial_step  = 12      # distance between "rings" of buses
angular_step = np.deg2rad(15)      # angular increment inside a block

coords = {}
rings  = {1: 0, 2: 0, 3: 0, 4: 0}  # keep track of how many buses are already placed in each block

for bus_idx, bus_row in net.bus.iterrows():
    # MATPOWER keeps the logical "area" in column 'zone'
    area = int(bus_row.zone) if 'zone' in bus_row else 1  # default to 1 if missing
    cx, cy = block_centres.get(area, (0, 0))

    # place buses in concentric rings so labels/lines don’t overlap
    ring   = rings[area] // 24          # new ring every 24 buses
    pos_in_ring = rings[area] % 24

    r   = radial_step * (1 + ring)
    ang = pos_in_ring * angular_step

    x = cx + r * np.cos(ang)
    y = cy + r * np.sin(ang)

    coords[bus_idx] = (x, y)
    rings[area] += 1

# assign to pandapower
net.bus_geodata = coords          # overwrite any existing (None) layout

# ------------------------------------------------------------------
# 3. quick static plot
# ------------------------------------------------------------------
pp.simple_plot(net,
               plot_loads=True,
               line_width=2,
               bus_size=0.05,
               show_plot=True)

#%%

import pandapower.plotting as pp
import pandapower.networks as pn
import plotly.graph_objects as go
import pandas as pd

# 1. get or build a network and run a power‑flow
net = pn.case118()       # any network works

# 2. collect bus positions
if not hasattr(net, "bus_geodata") or net.bus_geodata.empty:
    #net.bus_geodata = pd.DataFrame()  # ensure attribute exists
    pp.create_generic_coordinates(net)

pp.runpp(net)                 # fill res_* tables

geo = net.bus_geodata.reindex(net.bus.index)
x, y = geo.x.values, geo.y.values             # -> XY plane

# 3. choose Z‑value
z = net.res_bus.vm_pu.values                  # voltage magnitude as height

# 4a. buses: one Scatter3d trace
bus_trace = go.Scatter3d(
    x=x, y=y, z=z,
    mode="markers",
    marker=dict(size=5,
                color=z,            # color = same quantity or something else
                colorscale="Viridis",
                colorbar=dict(title="V [p.u.]")),
    name="buses"
)

# 4b. lines: one Scatter3d per line (to connect the dots)
line_traces = []
for _, ln in net.line.iterrows():
    fb, tb = ln.from_bus, ln.to_bus
    x_line = [geo.at[fb, 'x'], geo.at[tb, 'x'], None]
    y_line = [geo.at[fb, 'y'], geo.at[tb, 'y'], None]
    z_line = [z[fb],           z[tb],            None]   # same z‑metric at both ends

    line_traces.append(go.Scatter3d(
        x=x_line, y=y_line, z=z_line,
        mode="lines", line=dict(width=3, color="grey"),
        hoverinfo="none", showlegend=False)
    )

# 5. put everything together
fig = go.Figure(data=[bus_trace] + line_traces)
fig.update_layout(
    scene=dict(
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        zaxis_title="Voltage [p.u.]",
        aspectmode="data"),
    margin=dict(l=0, r=0, b=0, t=40),
    title="3‑D view of network voltages")
fig.show()

#%%
from pandapower.create import create_empty_network, create_bus, create_line, create_transformer3w, create_transformer, create_ext_grid, create_load
import pandapower.networks as nw
import pandapower.plotting.plotly as pplotly
from pandas import Series
import numpy as np

net = nw.case118()

lc = pplotly.create_line_trace(net,net.line.index, color='black')
bc = pplotly.create_bus_trace(net, net.bus.index, size=10, color="orange",
                              infofunc=Series(index=net.bus.index,
                                              data=net.bus.name + '<br>' + net.bus.vn_kv.astype(str) + ' kV'))
pplotly.draw_traces(bc + lc, figsize=1, aspectratio=(8,6));
