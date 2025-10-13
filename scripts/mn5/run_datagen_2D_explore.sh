#!/bin/bash
# Run batches of different configurations of datagen in slurm

export COMPSS_PYTHON_VERSION="3.12.1"
module load hdf5
module load sqlite3
module load python/3.12.1
module use /apps/GPP/modulefiles/applications/COMPSs/.custom
module load TrunkMauro

# Parse username splitting the string by delimiter "/"
IFS='/' read -ra parts <<< "$HOME"
# Get the last element of the array
username="${parts[-1]}"

# Set up variables and directories
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "${SCRIPT_DIR}"/../.. || exit
datagen_root_dir=$(pwd)
working_dir="/gpfs/scratch/bsc19/$username"
mkdir -p "${datagen_root_dir}/setup/tmp/"
export PYTHONPATH="${PYTHONPATH}:${datagen_root_dir}"

# Run application
export COMPUTING_UNITS=1
enqueue_compss \
  --pythonpath="${PYTHONPATH}" \
  --num_nodes=1 \
  --job_execution_dir="${datagen_root_dir}" \
  --worker_working_dir="${working_dir}" \
  --master_working_dir="${working_dir}" \
  --lang=python \
  --exec_time=30 \
  --tracing \
  --project_name=bsc19 \
  --qos=gp_debug \
  --log_dir="${working_dir}" \
  --agents \
  run_datagen_2D_explore.py "${datagen_root_dir}"
