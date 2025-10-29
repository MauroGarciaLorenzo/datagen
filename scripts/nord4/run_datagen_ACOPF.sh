module load hdf5
module load python/3.12.0
module use /apps/modules/modulefiles/tools/COMPSs/.custom
module load TrunkMauroNord4

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
yaml_file="${datagen_root_dir}/setup/default_setup.yaml"
working_dir="/gpfs/scratch/bsc19/$username"
export PYTHONPATH="${datagen_root_dir}/packages/:${PYTHONPATH}:${datagen_root_dir}"

# Print user information
echo "Username is: $username"
echo "Using $working_dir as the working directory"
echo "Current directory is $(pwd)"

# Run COMPSs execution
num_nodes=2
computing_units=5
while [ ${num_nodes} -le 2 ]
do
  export COMPUTING_UNITS=${computing_units}
  enqueue_compss \
  --pythonpath=${PYTHONPATH} \
  --num_nodes=${num_nodes} \
  --job_execution_dir="${datagen_root_dir}" \
  --worker_working_dir=${working_dir} \
  --master_working_dir=${working_dir} \
  --lang=python \
  --exec_time=120 \
  --tracing \
  --project_name=bsc19 \
  --qos=gp_debug \
  --log_dir=${working_dir} \
  --agents \
  run_datagen_ACOPF.py "${datagen_root_dir}" "${input_data}" "${yaml_file}"

  num_nodes=$((num_nodes * 2))
done
