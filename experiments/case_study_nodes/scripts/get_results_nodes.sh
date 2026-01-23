#!/bin/bash

# Parse username splitting the string by delimiter "/"
IFS='/' read -ra parts <<< "$HOME"
username="${parts[-1]}"

# Set up variables and directories
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "${SCRIPT_DIR}"/../.. || exit
results_dir="${SCRIPT_DIR}/../results/raw"
results_path="${results_dir}/results.csv"
logs_dir="${SCRIPT_DIR}/../../../results"

# Ensure results directory exists
mkdir -p "$results_dir"

# Print user information
echo "Username is: $username"
echo "Results path is: $results_path"

num_nodes=1
# Run COMPSs execution
while [ ${num_nodes} -le 64 ]
do
  sacct --name=nodes${num_nodes}_case_study --starttime=2025-03-5 \
  --format=JobID,JobName,Elapsed,NNodes,AllocCPUS,State,ExitCode -P \
  | awk -F'|' -v logs_dir="$logs_dir" '
    BEGIN { OFS="|" }
    $6=="COMPLETED" && $2 ~ /^nodes[0-9]+_case_study$/ {
        job_id=$1
        n_cases=0
        execution_time=""
        
        # Find the job directory
        cmd="find " logs_dir " -type d -name \"*" job_id "*\" | head -n 1"
        cmd | getline job_dir
        close(cmd)


        if (job_dir != "") {
            # Get the number of cases from cases_df.csv
            cases_file=job_dir "/cases_df.csv"
            cmd="test -f \"" cases_file "\" && echo EXISTS || echo MISSING"
            cmd | getline cases_exists
            close(cmd)


            if (cases_exists == "EXISTS") {
                cmd="wc -l < \"" cases_file "\""
                cmd | getline n_cases
                close(cmd)
            }

            # Get the execution time from execution_time.csv
            execution_time_file=job_dir "/execution_time.csv"
            cmd="test -f \"" execution_time_file "\" && echo EXISTS || echo MISSING"
            cmd | getline execution_exists
            close(cmd)


            if (execution_exists == "EXISTS") {
                cmd="tail -n +2 \"" execution_time_file "\" | head -n 1"
                cmd | getline execution_time
                close(cmd)
            }

        }
        # Ensure execution_time is set correctly
        if (execution_time == "") {
            execution_time = "UNKNOWN"
        }


        print $1 "|" $2 "|" execution_time "|" $4 "|" $5 "|" $6 "|" $7 "|" n_cases
    }' >> "$results_path"

  num_nodes=$((num_nodes * 2))
done

