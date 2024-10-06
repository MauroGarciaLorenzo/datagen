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
cd ../..
datagen_root_dir=$(pwd)
yaml_file="${datagen_root_dir}/setup/setup_example.yaml"
results_dir="/gpfs/scratch/bsc19/$username"
export PYTHONPATH="${datagen_root_dir}/packages/:${PYTHONPATH}:${datagen_root_dir}"

# Print user information
echo "Username is: $username"
echo "Using $results_dir as the results directory"
echo "Current directory is $(pwd)"

# Run COMPSs execution
num_nodes=1
computing_units=1
while [ ${num_nodes} -le 1 ]
do
  export COMPUTING_UNITS=${computing_units}
  enqueue_compss \
  --pythonpath=${PYTHONPATH} \
  --num_nodes=${num_nodes} \
  --job_execution_dir="${datagen_root_dir}" \
  --worker_working_dir=${results_dir} \
  --master_working_dir=${results_dir} \
  --lang=python \
  --exec_time=1000 \
  --tracing \
  --project_name=bsc19 \
  --qos=gp_bsccs \
  --log_dir=${results_dir} \
  -d \
  --agents \
  run_datagen_ACOPF.py --results_dir="${datagen_root_dir}" "${yaml_file}"

  num_nodes=$((num_nodes * 2))
done
