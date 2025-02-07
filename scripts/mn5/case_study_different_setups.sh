#!/bin/bash
# Run batches of different configurations of datagen in slurm

export COMPSS_PYTHON_VERSION="3.12.1"
module load hdf5
module load sqlite3
module load python/3.12.1
module load COMPSs/3.3

#
# Define the cases to run as a 2D array
#
# ORDER: n_samples | n_cases | max_depth | seed
cases=(
  "5 5 5 17"
  "5 5 5 28"
  "5 5 5 70"
  "5 5 20 17"
  "5 5 50 17"
  "10 10 5 17"
)

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
base_yaml_file="${datagen_root_dir}/setup/default_setup.yaml"
working_dir="/gpfs/scratch/bsc19/$username"
mkdir -p "${datagen_root_dir}/setup/tmp/"
export PYTHONPATH="${datagen_root_dir}/packages/:${PYTHONPATH}:${datagen_root_dir}"

# Path to your YAML file
if [ ! -f "$base_yaml_file" ]; then
  echo "Error: YAML file not found at $base_yaml_file"
  exit 1
fi

# Loop over each case
for case in "${cases[@]}"; do
  # Read the parameters from the current case
  read -r n_samples n_cases max_depth seed <<< "$case"

  # Call your function or command
  echo -e "\n============"
  echo -e "Running case"
  echo -e "============\n"
  echo "n_samples: $n_samples"
  echo "n_cases: $n_cases"
  echo "max_depth: $max_depth"
  echo "seed: $seed"

  # Set up a config file for each run
  yaml_file="${datagen_root_dir}/setup/tmp/setup_mn5_${n_samples}${n_cases}${max_depth}${seed}.yaml"
  cp $base_yaml_file $yaml_file

  # Use sed to update the values in the YAML file
  # MAKE SURE THE YAML PARAMETERS ARE CORRECTLY INDENTED, OTHERWISE SED WON'T WORK
  sed -i "s/^n_samples: .*/n_samples: $n_samples/" "$yaml_file"
  sed -i "s/^n_cases: .*/n_cases: $n_cases/" "$yaml_file"
  sed -i "s/^max_depth: .*/max_depth: $max_depth/" "$yaml_file"
  sed -i "s/^seed: .*/seed: $seed/" "$yaml_file"

  # Run application
  export COMPUTING_UNITS=8
  enqueue_compss \
    --pythonpath="${PYTHONPATH}" \
    --num_nodes=1 \
    --job_execution_dir="${datagen_root_dir}" \
    --worker_working_dir="${working_dir}" \
    --master_working_dir="${working_dir}" \
    --lang=python \
    --exec_time=2880 \
    --tracing \
    --project_name=bsc19 \
    --qos=gp_bsccs \
    --log_dir="${working_dir}" \
    -d \
    --agents \
    run_datagen_ACOPF.py "${datagen_root_dir}" "${input_data}" "${yaml_file}"
done
