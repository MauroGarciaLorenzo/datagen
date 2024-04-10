module load python/3.9.10
pip install scipy
module load COMPSs/3.3

cd ..
echo $(pwd)

export PYTHONPATH="$(pwd)/.."

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
  --exec_time=15 \
  --agents \
  --tracing \
  run_N_PF.py "${current_directory}/../results/run_N_PF_${num_nodes}"
  num_nodes=$((num_nodes * 2))
done
