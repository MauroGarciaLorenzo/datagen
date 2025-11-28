#!/bin/bash
# Execute the whole data generator in a large sized run
# Produces 127,500 or 91,500 depending on whether using entropy or not
#

export COMPSS_PYTHON_VERSION="3.12.1"
module load hdf5
module load sqlite3
module load python/3.12.1
module load COMPSs/Trunk

# Variables initialization
num_nodes=8
yaml_file="setup_seed3_nc3_ns333_d5_cpus8_th0.2_fr0.05_sensitivity.yaml"

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
yaml_path="${datagen_root_dir}/setup/${yaml_file}"
working_dir="/gpfs/scratch/bsc19/$username"
export PYTHONPATH="${datagen_root_dir}/packages/:${PYTHONPATH}:${datagen_root_dir}"

# Print user information
echo "Username is: $username"
echo "Using $working_dir as the working directory"
echo "Current directory is $(pwd)"

# Extract COMPUTING_UNITS from YAML
COMPUTING_UNITS=$(grep -A 5 '^environment:' "$yaml_path" | grep 'COMPUTING_UNITS' | awk '{print $2}' | tr -d ' ')

if [[ -z "$COMPUTING_UNITS" ]]; then
  echo "COMPUTING_UNITS not found in YAML"
  exit 1
fi
echo "Using ${COMPUTING_UNITS} computing units"
export COMPUTING_UNITS

# Run COMPSs execution
enqueue_compss \
--pythonpath=${PYTHONPATH} \
--num_nodes=${num_nodes} \
--job_name="datagen_cu${computing_units}_nodes${num_nodes}" \
--job_execution_dir="${datagen_root_dir}" \
--worker_working_dir=${working_dir} \
--master_working_dir=${working_dir} \
--lang=python \
--exec_time=2880 \
--tracing \
--project_name=bsc19 \
--qos=gp_bsccs \
--log_dir=${working_dir} \
--agents \
run_datagen_ACOPF.py "${datagen_root_dir}" "${input_data}" "${yaml_path}"
