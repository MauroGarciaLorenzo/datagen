import os
import random
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from datagen.src.objective_function import complex_2d_shape_obj_func, \
    complex_2d_shape
from datagen.src.dimensions import Dimension
from datagen.src.start_app import start

try:
    from pycompss.api.task import task
    from pycompss.api.api import compss_wait_on
except ImportError:
    from datagen.dummies.task import task
    from datagen.dummies.api import compss_wait_on


@task(on_failure='FAIL')
def main(working_dir):
    n_samples = 20
    n_cases = 1
    rel_tolerance = 0.02
    max_depth = 10
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    dimensions = [
        Dimension(label="tau_Dim_0", n_cases=n_cases, divs=2, borders=(-1, 1)),
        Dimension(label="tau_Dim_1", n_cases=n_cases, divs=2, borders=(-1, 1))
    ]
    use_sensitivity = False
    seed = 17
    divs_per_cell = 3
    feasible_rate = 0.5
    entropy_threshold = 0.2
    delta_entropy_threshold = 0
    chunk_size = 5000

    # Get computing units assigned to the objective function
    cu = os.environ.get("COMPUTING_UNITS", default=None)
    cu_str = ""
    if cu:
        cu_str = f"_cu{cu}"
    print("COMPUTING_UNITS: ", cu)
    # Get slurm job id
    slurm_job_id = os.getenv("SLURM_JOB_ID", default=None)
    slurm_str = ""
    if slurm_job_id:
        slurm_str = f"_slurm{slurm_job_id}"
    # Get slurm n_nodes
    slurm_num_nodes = os.environ.get('SLURM_JOB_NUM_NODES', default=None)
    slurm_nodes_str = ""
    if slurm_num_nodes:
        slurm_nodes_str = f"_nodes{slurm_num_nodes}"
    print("NUMBER OF NODES: ", slurm_num_nodes)
    # Create unique directory name for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rnd_num = random.randint(1000, 9999)
    dir_name = f"datagen_2D_explore{slurm_str}{cu_str}{slurm_nodes_str}_seed{seed}_nc{n_cases}" \
               f"_ns{n_samples}_d{max_depth}_{timestamp}_{rnd_num}"
    path_results = os.path.join(
        working_dir, "results", dir_name)
    if not os.path.isdir(path_results):
        os.makedirs(path_results)

    # Plot heat map
    add_color_map(ax, fig)

    cases_df, dims_df, execution_logs, output_dataframes = \
        start(dimensions, n_samples, rel_tolerance,
              func=complex_2d_shape_obj_func,
              max_depth=max_depth, dst_dir=path_results,
              use_sensitivity=use_sensitivity, ax=ax,
              divs_per_cell=divs_per_cell, plot_boxplot=True, seed=seed,
              feasible_rate=feasible_rate, chunk_size=chunk_size,
              entropy_threshold=entropy_threshold,
              delta_entropy_threshold=delta_entropy_threshold)

    fig_path = os.path.join(
        path_results, "figures/2D_contour_plot_complex_shape.png")
    fig.savefig(fig_path, dpi=300)
    plt.show()


def add_color_map(ax, fig):
    # Generate the data grid
    x = np.linspace(-3, 3, 500)  # 500 points in range [-3, 3]
    y = np.linspace(-3, 3, 500)
    X, Y = np.meshgrid(x, y)  # Create a 2D grid

    # Compute Z values
    Z = complex_2d_shape(X, Y)

    # Plot the filled contour plot (heatmap)
    c = ax.contourf(X, Y, Z, levels=100, cmap='viridis')

    # Add contour lines for z = 0
    zero_contour = ax.contour(X, Y, Z, levels=[0], colors='red',
                              linewidths=1.5)
    ax.clabel(zero_contour, fmt="z=0", colors="white", fontsize=10)

    fig.colorbar(c, ax=ax, label='Z value')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Heatmap of complex_2d_shape')


if __name__ == '__main__':
    results_dir = os.path.join(os.getcwd(), "results")
    main(results_dir)
