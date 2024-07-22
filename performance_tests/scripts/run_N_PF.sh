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
cd ..
current_dir=$(pwd)
datagen_root_dir="${current_dir}/.."
stability_dir="${datagen_root_dir}/../stability_analysis"
input_data="${stability_dir}/stability_analysis/data"
yaml_file="${datagen_root_dir}/setup/default_setup.yaml"
working_dir="/gpfs/scratch/bsc19/$username"
export PYTHONPATH="${current_dir}/../packages/:${PYTHONPATH}:${current_dir}/../:${current_dir}"

# Print user information
echo "Username is: $username"
echo "Using $working_dir as the working directory"
echo "Current directory is ${current_dir}"

# Run COMPSs execution
num_nodes=1
computing_units=1
while [ ${num_nodes} -le 1 ]
do
  export COMPUTING_UNITS=${computing_units}
  enqueue_compss \
  --pythonpath=${PYTHONPATH} \
  --num_nodes=${num_nodes} \
  --job_execution_dir="${current_dir}/.." \
  --worker_working_dir=${working_dir} \
  --master_working_dir=${working_dir} \
  --lang=python \
  --exec_time=1000 \
  --tracing \
  --project_name=bsc19 \
  --qos=gp_bsccs \
  --log_dir=${working_dir} \
  -d \
  --agents \
  run_N_PF.py --working_dir="${current_dir}" --path_data="${input_data}" --setup="${yaml_file}"

  num_nodes=$((num_nodes * 2))
done
