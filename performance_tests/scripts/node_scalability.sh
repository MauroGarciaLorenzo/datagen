module load python/3.9.10
pip install scipy
module load COMPSs/3.2

num_nodes=1

while [ ${num_nodes} -le 4 ]
do
  enqueue_compss \
  --pythonpath="../"  \
  --num_nodes=${num_nodes} \
  --worker_working_dir=local_disk \
  --master_working_dir=local_disk \
  --lang=python \
  --qos=debug \
  --exec_time=10 \
  --agents \
  -t \
  node_scalability_test
  num_nodes=$((num_nodes * 2))
done
