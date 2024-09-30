#!/bin/bash

# Go to the root of the repository
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/../.." || exit

# Set up the environment (Make sure to set up a Python venv environment with all dependencies and COMPSs)
source venv/bin/activate

# Set up variables and directories
datagen_root_dir=$(pwd)
stability_dir="${datagen_root_dir}/../stability_analysis"
input_data="${stability_dir}/stability_analysis/data"

# Logging configuration
mkdir logs
app_log_file="${datagen_root_dir}/logs/log_test_app_compss.txt"
compss_log_dir="$HOME/logs"
rm ${app_log_file}

# Path to your YAML file
yaml_file="${datagen_root_dir}/setup/test_setup.yaml"
if [ ! -f "$yaml_file" ]; then
  echo "Error: YAML file not found at $yaml_file" | tee -a ${app_log_file}
  exit 1
fi

# Loop through all combinations of max_depth, n_cases, and n_samples
failed_runs=""
for max_depth in 1 2 3; do
  for n_cases in 1 2 3; do
    for n_samples in 1 2 3; do
      echo -e "\n=======================================================================" | tee -a ${app_log_file}
      echo -e "=== Running case with max_depth=$max_depth, n_cases=$n_cases, n_samples=$n_samples" | tee -a ${app_log_file}
      echo -e "=======================================================================\n" | tee -a ${app_log_file}

      # Use sed to update the values in the YAML file
      # MAKE SURE THE YAML PARAMETERS ARE CORRECTLY INDENTED, OTHERWISE SED WON'T WORK
      sed -i "s/^max_depth: .*/max_depth: $max_depth/" "$yaml_file"
      sed -i "s/^n_cases: .*/n_cases: $n_cases/" "$yaml_file"
      sed -i "s/^n_samples: .*/n_samples: $n_samples/" "$yaml_file"

      # Clean up COMPSs processes
      compss_clean_procs

      # Run the test
      timeout 20m compss_agent_start_service \
        --lang=PYTHON \
        --num_agents=3 \
        --hostname=localhost \
        --method_name=main \
        -d \
        -t \
        --topology=tree \
        --log_dir="$compss_log_dir" \
        --verbose \
        run_datagen_ACOPF "$(pwd)" "${input_data}" "${yaml_file}" 2>&1 | tee -a "$app_log_file"

      # Check exit status
      if [ "${PIPESTATUS[0]}" -eq 124 ]; then
        echo -e "\n=== The run timed out ===" | tee -a "$app_log_file"
        failed_runs+="\nmax_depth=$max_depth, n_cases=$n_cases, n_samples=$n_samples"
        # Find the first file with "RESUBMITTED" in its name and ending with ".err"
        file=$(find "$compss_log_dir" -type f -path '*/jobs/*' -name '*RESUBMITTED*.err' | head -n 1)
        if [ -n "$file" ]; then
            # Append the content of the found file to the log file
            cat "$file"
            cat "$file" >> "$app_log_file"
            error_msg="\n   Found potential error file in $file"
        else
            error_msg="\n   No file with 'RESUBMITTED' in the name and ending with '.err' found."
        fi
        echo -e "$error_msg" | tee -a "$app_log_file"
        failed_runs+="$error_msg"
      else
        echo -e "\n=== Finished run with max_depth=$max_depth, n_cases=$n_cases, n_samples=$n_samples ===\n" | tee -a "$app_log_file"
      fi

    done
  done
done

# Clean up COMPSs processes
compss_clean_procs

echo -e "\n\n\n=== ALL RUNS FINISHED ===" | tee -a "$app_log_file"
echo -e "=== Failed runs ===" | tee -a "$app_log_file"
echo -e "$failed_runs" | tee -a "$app_log_file"
