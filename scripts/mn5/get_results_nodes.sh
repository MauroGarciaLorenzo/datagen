#!/bin/bash

# Parse username splitting the string by delimiter "/"
IFS='/' read -ra parts <<< "$HOME"
username="${parts[-1]}"

# Set up variables and directories
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "${SCRIPT_DIR}"/../.. || exit
results_path="/gpfs/scratch/bsc19/${username}/tests/case_study_nodes/results/results.csv"
results_dir="${SCRIPT_DIR}/../../results"

# Create the directory for the results file if it doesn't exist
mkdir -p "$(dirname "$results_path")"

# Create the results file if it doesn't exist
if [ ! -f "$results_path" ]; then
    echo "JobID|JobName|Elapsed|NNodes|AllocCPUS|State|ExitCode|n_cases" > "$results_path"
fi

# Print user information
echo "Username is: $username"
echo "Results path is: $results_path"

num_nodes=1
# Run COMPSs execution
while [ ${num_nodes} -le 64 ]
do
  sacct --name=nodes${num_nodes}_case_study --starttime=2025-03-5 \
  --format=JobID,JobName,Elapsed,NNodes,AllocCPUS,State,ExitCode -P \
  | awk -F'|' -v results_dir="$results_dir" '
    BEGIN { OFS="|" }
    $6=="COMPLETED" && $2 ~ /^nodes[0-9]+_case_study$/ {
        job_id=$1
        n_cases=0
        cmd="find " results_dir " -type d -name \"*" job_id "*\" | head -n 1"
        cmd | getline job_dir
        close(cmd)
        if (job_dir != "") {
            cases_file=job_dir "/cases_df.csv"
            cmd="wc -l < " cases_file
            cmd | getline n_cases
            close(cmd)
        }
        print $0, n_cases
    }' >> "$results_path"

  num_nodes=$((num_nodes * 2))
done
