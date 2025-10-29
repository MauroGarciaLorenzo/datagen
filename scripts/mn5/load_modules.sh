PATH_DATAGEN="$1"
PATH_STABILITY_ANALYSIS="$2"
PATH_GRIDCAL="$3"
MACHINE_NODE="$4"

echo "LOAD MODULES&&&&&&&&&&&&&&&&&&&&&&&&"

export COMPSS_PYTHON_VERSION="3.12.1"
module load hdf5
module load sqlite3
module load python/3.12.1
module use /apps/GPP/modulefiles/applications/COMPSs/.custom
module load TrunkMauro

export PYTHONPATH="${PATH_DATAGEN}/packages/:${PYTHONPATH}:${PATH_DATAGEN}"

if [ $MACHINE_NODE == "glogin4" ];then
  pip install -r "${PATH_DATAGEN}/requirements.txt" --target="${PATH_DATAGEN}/packages/"
  pip install -e "${PATH_GRIDCAL}/src/GridCalEngine"
  pip install -e "${PATH_STABILITY_ANALYSIS}"
fi
