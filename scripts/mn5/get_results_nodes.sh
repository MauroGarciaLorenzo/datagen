#!/bin/bash

# Parse username splitting the string by delimiter "/"
IFS='/' read -ra parts <<< "$HOME"
username="${parts[-1]}"

# Set up variables and directories
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "${SCRIPT_DIR}"/../.. || exit
results_path="/gpfs/scratch/bsc19/${username}/tests/case_study_nodes/results/results.csv"

# Create the directory for the results file if it doesn't exist
mkdir -p "$(dirname "$results_path")"

# Create the results file if it doesn't exist
if [ ! -f "$results_path" ]; then
    touch "$results_path"
fi

# Print user information
echo "Username is: $username"
echo "Results path is: $results_path"

num_nodes=1
# Run COMPSs execution
while [ ${num_nodes} -le 64 ]
do
  sacct --name=nodes${num_nodes}_case_study --starttime=2025-02-25 \
  --format=JobID,JobName,Elapsed,NNodes,AllocCPUS,State,ExitCode -P \
  | awk -F'|' '$6=="COMPLETED" && $2 ~ /^nodes[0-9]+_case_study$/' >> "$results_path"

  num_nodes=$((num_nodes * 2))
done

pwd
cd scripts/postprocess || exit
python plot_results_scalability.py $results_path

