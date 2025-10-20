module load python/3.9.10
pip install scipy
module use /apps/GPP/modulefiles/applications/COMPSs/.custom
module load TrunkMauro

current_directory=$(pwd)

echo "$current_directory"

export PYTHONPATH="${current_directory}/..":"${current_directory}":"${current_directory}/../.."

num_nodes=1

while [ ${num_nodes} -le 8 ]
do
  enqueue_compss \
  --pythonpath=${PYTHONPATH} \
  --job_execution_dir="${current_directory}/.." \
  --num_nodes=${num_nodes} \
  --worker_working_dir=local_disk \
  --master_working_dir=local_disk \
  --lang=python \
  --exec_time=400 \
  --agents \
  --tracing \
  node_scalability_test.py "${current_directory}/../results/node_scalability${num_nodes}"
  num_nodes=$((num_nodes * 2))
done
