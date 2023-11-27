module load python/3.9.10
pip install scipy
module load COMPSs/3.3

current_directory=$(pwd)

echo "$current_directory"

export PYTHONPATH="/home/bsc19/bsc19019/hp2c-dt/datagen/performance_tests":"/home/bsc19/bsc19019/hp2c-dt/datagen/performance_tests/scripts":"/home/bsc19/bsc19019/hp2c-dt/datagen"

num_nodes=1

while [ ${num_nodes} -le 8 ]
do
  enqueue_compss \
  --pythonpath=${PYTHONPATH} \
  --job_execution_dir="/home/bsc19/bsc19019/hp2c-dt/datagen/performance_tests" \
  --num_nodes=${num_nodes} \
  --worker_working_dir=local_disk \
  --master_working_dir=local_disk \
  --lang=python \
  --exec_time=400 \
  --agents \
  --tracing \
  node_scalability_test.py
  num_nodes=$((num_nodes * 2))
done
