#!/bin/bash
# Execute the whole data generator in a medium sized run with 1 node and
# changing the number computing units used in the objective function to
# evaluate which configuration is better globally.

export COMPSS_PYTHON_VERSION="3.12.1"
module load hdf5
module load sqlite3
module load python/3.12.1
module load COMPSs/3.3

# Parse username splitting the string by delimiter "/"
IFS='/' read -ra parts <<< "$HOME"
# Get the last element of the array
username="${parts[-1]}"

# Set up variables and directories
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "${SCRIPT_DIR}"/../.. || exit
datagen_root_dir=$(pwd)
stability_dir="${datagen_root_dir}/../stability_analysis"
input_data="${stability_dir}/stability_analysis/data"
yaml_file="${datagen_root_dir}/setup/setup_seed17_nc5_ns5_d5.yaml"
working_dir="/gpfs/scratch/bsc19/$username"
export PYTHONPATH="${datagen_root_dir}/packages/:${PYTHONPATH}:${datagen_root_dir}"

# Print user information
echo "Username is: $username"
echo "Using $working_dir as the working directory"
echo "Current directory is $(pwd)"

# Variables initialization
num_nodes=1
computing_units=1

# Run COMPSs execution
while [ ${computing_units} -le 112 ]
do
  # Assign the max number of computing units used by the objective function
  export COMPUTING_UNITS=${computing_units}
  enqueue_compss \
  --pythonpath=${PYTHONPATH} \
  --num_nodes=${num_nodes} \
  --job_execution_dir="${datagen_root_dir}" \
  --worker_working_dir=${working_dir} \
  --master_working_dir=${working_dir} \
  --lang=python \
  --exec_time=1440 \
  --tracing \
  --project_name=bsc19 \
  --qos=gp_bsccs \
  --log_dir=${working_dir} \
  -d \
  --agents \
  run_datagen_ACOPF.py "${datagen_root_dir}" "${input_data}" "${yaml_file}"

  if [ ${computing_units} -eq 64 ]; then
    computing_units=112
  else
    computing_units=$((computing_units * 2))
  fi
done
