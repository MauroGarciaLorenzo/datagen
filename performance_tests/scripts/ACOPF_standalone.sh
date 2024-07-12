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
export PYTHONPATH="${current_directory}/../packages/:${PYTHONPATH}:${current_directory}/../../:${current_directory}/../:${current_directory}"

# Print user information
echo "Username is: $username"
echo "Using $working_dir as the working directory"
echo "Current directory is ${current_directory}"

# Run COMPSs execution
computing_units=1
while [ ${computing_units} -le 112 ]
do
  export COMPUTING_UNITS=${computing_units}
  enqueue_compss \
  --pythonpath=${PYTHONPATH} \
  --num_nodes=1 \
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
  ACOPF_standalone.py "$PWD"
  if [ ${computing_units} -eq 64 ]; then
    computing_units=112
  fi
  computing_units=$((computing_units * 2))
done
