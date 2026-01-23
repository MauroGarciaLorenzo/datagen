#!/bin/bash

# Default values
log_dir="/gpfs/scratch/bsc19/bsc019019/.COMPSs"
job_id=""

# Parse arguments
for arg in "$@"; do
    case $arg in
        --log_dir=*)
            log_dir="${arg#*=}"
            ;;
        --job_id=*)
            job_id="${arg#*=}"
            ;;
        *)
            echo "Unknown parameter passed: $arg"
            exit 1
            ;;
    esac
done

# Validate job_id
if [ -z "$job_id" ]; then
    echo "Error: --job_id must be specified"
    exit 1
fi

target_dir="$log_dir/$job_id"

# Check if directory exists
if [ ! -d "$target_dir" ]; then
    echo "Directory does not exist: $target_dir"
    exit 1
fi

tmp_file=$(mktemp)
grep -ri "Depth=" "$target_dir" > "$tmp_file"

awk -F'[:,=]' '
{
    for (i = 1; i <= NF; i++) {
        if ($i ~ /Depth/) d=$(i+1);
        if ($i ~ /Entropy/ && $(i-1) != "Delta") e=$(i+1);
        if ($i ~ /Delta_entropy/) de=$(i+1);
    }
    count[d]++
    sum_e[d] += e
    sum_de[d] += de
    if (!(d in max_e) || e > max_e[d]) max_e[d] = e
}
END {
    printf "%5s %15s %20s %15s\n", "Depth", "Avg Entropy", "Avg Delta Entropy", "Max Entropy";
    PROCINFO["sorted_in"] = "@ind_num_asc";
    for (d in count) {
        printf "%5d %15.6f %20.6f %15.6f\n", d, sum_e[d]/count[d], sum_de[d]/count[d], max_e[d];
    }
}' "$tmp_file"

rm "$tmp_file"
