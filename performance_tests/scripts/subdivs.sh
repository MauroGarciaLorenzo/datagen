module load python/3.9.10
pip install scipy
module use /apps/GPP/modulefiles/applications/COMPSs/.custom
module load COMPSs/Trunk

current_directory=$(pwd)

echo "$current_directory"

export PYTHONPATH="/home/bsc19/bsc19019/hp2c-dt/datagen/performance_tests":"/home/bsc19/bsc19019/hp2c-dt/datagen/performance_tests/scripts":"/home/bsc19/bsc19019/hp2c-dt/datagen"

num_nodes=1

while [ ${num_nodes} -le 4 ]
do
  enqueue_compss \
  --pythonpath=${PYTHONPATH} \
  --job_execution_dir="/home/bsc19/bsc19019/hp2c-dt/datagen/performance_tests" \
  --num_nodes=${num_nodes} \
  --worker_working_dir=local_disk \
  --master_working_dir=local_disk \
  --lang=python \
  --qos=debug \
  --exec_time=10 \
  --agents \
  --tracing \
  node_scalability_test.py
  num_nodes=$((num_nodes * 2))
done
