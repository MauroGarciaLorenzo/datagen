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
current_directory=$(pwd)
working_dir="/gpfs/scratch/bsc19/$username"
export PYTHONPATH="${current_directory}/../packages/:${PYTHONPATH}:${current_directory}/../:${current_directory}"

# Print user information
echo "Username is: $username"
echo "Using $working_dir as the working directory"
echo "Current directory is ${current_directory}"

# Run COMPSs execution
num_nodes=1
while [ ${num_nodes} -le 1 ]
do
  enqueue_compss \
  --pythonpath=${PYTHONPATH} \
  --num_nodes=${num_nodes} \
  --job_execution_dir="${current_directory}/.." \
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
  run_N_PF.py "$PWD"

  num_nodes=$((num_nodes * 2))
done
