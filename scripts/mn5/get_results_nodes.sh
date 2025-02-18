#!/bin/bash

# Parse username splitting the string by delimiter "/"
IFS='/' read -ra parts <<< "$HOME"
# Get the last element of the array
username="${parts[-1]}"

# Set up variables and directories
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "${SCRIPT_DIR}"/../.. || exit
results_path="/gpfs/scratch/bsc19/${username}/tests/case_study_nodes/results/results.csv"

# Print user information
echo "Username is: $username"
echo "Results path is: $results_path"

num_nodes=1
# Run COMPSs execution
while [ ${num_nodes} -le 64 ]
do
  sacct --name=nodes${num_nodes}_case_study --starttime=2025-02-17 --format=JobID,JobName,Elapsed,NNodes,AllocCPUS,State,ExitCode -P >> $results_path

  num_nodes=$((num_nodes * 2))
done
