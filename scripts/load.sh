#!/bin/bash

# Check if the correct number of arguments are provided
if [[ $# -ne 2 ]]; then
    echo "Usage: ./load.sh <path_datagen> <path_stability_analysis> <path_gridcal> <machine>"
    exit 1
fi

# Assign arguments to variables
PATH_DATAGEN="$1"
PATH_STABILITY_ANALYSIS="$2"
PATH_GRIDCAL="$3"
MACHINE="$4"
MACHINE_NODE="$5"

# Assuming PATH_DATAGEN contains the path with or without a trailing slash
if [[ $PATH_DATAGEN == */ ]]; then
    PATH_DATAGEN="${PATH_DATAGEN%/}"
fi

chmod +x "$PATH_DATAGEN/scripts/mn5/load_modules.sh"
chmod +x "$PATH_DATAGEN/scripts/nord/load_modules.sh"
# shellcheck disable=SC1090
source $PATH_DATAGEN/scripts/$MACHINE/load_modules.sh $PATH_DATAGEN $PATH_STABILITY_ANALYSIS $PATH_GRIDCAL $MACHINE_NODE
