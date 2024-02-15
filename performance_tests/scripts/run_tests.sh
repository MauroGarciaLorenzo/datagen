#!/bin/bash

usage() {
    echo "Usage: $0 [--min_nodes MIN_NODES] [--max_nodes MAX_NODES] [--test_name TEST_NAME]" 1>&2
    exit 1
}

# Default values
min_nodes=1
max_nodes=8

# Parse arguments in command line
while [[ $# -gt 0 ]]; do
    case "$1" in
        --min_nodes)
            min_nodes=$2
            shift 2
            ;;
        --max_nodes)
            max_nodes=$2
            shift 2
            ;;
        --test_name)
            test_name=$2
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

# Load modules
module load python/3.9.10
pip install scipy
module load COMPSs/3.3

current_directory=$(pwd)
echo "$current_directory"

export PYTHONPATH="${current_directory}/..":"${current_directory}":"${current_directory}/../.."

job_name=$(date "+%d%H%M")

# Run test for each num_nodes
while [ "$min_nodes" -le "$max_nodes" ]; do
    enqueue_compss \
    --pythonpath=${PYTHONPATH} \
    --job_execution_dir="${current_directory}/.." \
    --num_nodes=${min_nodes} \
    --worker_working_dir=local_disk \
    --master_working_dir=local_disk \
    --lang=python \
    --exec_time=120 \
    --agents=plain \
    --tracing \
    --debug \
    --qos=debug \
    --job_name="${job_name}_${min_nodes}" \
    --scheduler="es.bsc.compss.scheduler.orderstrict.fifo.FifoTS" \
    "${test_name}_test.py" "${current_directory}/../results/${job_name}/${test_name}/${min_nodes}"
    min_nodes=$((min_nodes * 2))
done
