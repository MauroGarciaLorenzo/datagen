USER=$1
NAMEFILE=$2
SETUP_FOLDER=$3
EXECUTION_FOLDER=$4
NUM_NODES=$5
EXEC_TIME=$6
QOS=$7
ROOT_DIR=$8
BRANCH=$9
DATA_DIR=${10}
gOPTION=${11}
tOPTION=${12}
dOPTION=${13}
PROJECT_NAME=${14}
NAME_SIM=${15}

echo "USER: $USER"
echo "NAMEFILE: $NAMEFILE"
echo "SETUP_FOLDER: $SETUP_FOLDER"
echo "EXECUTION_FOLDER: $EXECUTION_FOLDER"
echo "NUM_NODES: $NUM_NODES"
echo "EXEC_TIME: $EXEC_TIME"
echo "QOS: $QOS"
echo "ROOT_DIR: $ROOT_DIR"
echo "BRANCH: $BRANCH"
echo "DATA_DIR: $DATA_DIR"
echo "gOPTION: $gOPTION"
echo "tOPTION: $tOPTION"
echo "dOPTION: $dOPTION"
echo "PROJECT_NAME: $PROJECT_NAME"

if [[ "$EXECUTION_FOLDER" != /* ]]; then
  EXECUTION_FOLDER="$HOME/$EXECUTION_FOLDER"
fi

if [[ "$SETUP_FOLDER" != /* ]]; then
  SETUP_FOLDER="$HOME/$SETUP_FOLDER"
fi

if [[ "$DATA_DIR" != /* ]]; then
  DATA_DIR="$HOME/$DATA_DIR"
fi

# shellcheck disable=SC2164
cd $EXECUTION_FOLDER

# Construct the enqueue_compss command based on user options
enqueue_compss_cmd="enqueue_compss --project_name=$PROJECT_NAME --keep_workingdir --job_name=$NAME_SIM --agents --scheduler=es.bsc.compss.scheduler.orderstrict.fifo.FifoTS --job_execution_dir=$EXECUTION_FOLDER --log_dir=$EXECUTION_FOLDER --qos=$QOS --exec_time=$EXEC_TIME --pythonpath=$PYTHONPATH --num_nodes=$NUM_NODES --worker_in_master_cpus=112"

# Add -g option if specified
if [ "$gOPTION" = "true" ]; then
  enqueue_compss_cmd="$enqueue_compss_cmd -g"
fi

# Add -t option if specified
if [ "$tOPTION" = "true" ]; then
  enqueue_compss_cmd="$enqueue_compss_cmd -t"
fi

# Add -d option if specified
if [ "$dOPTION" = "true" ]; then
  enqueue_compss_cmd="$enqueue_compss_cmd -d"
fi

# Add the rest of the command
enqueue_compss_cmd="$enqueue_compss_cmd $ROOT_DIR/ACOPF_standalone.py --setup=$SETUP_FOLDER/$NAMEFILE --working_dir=$EXECUTION_FOLDER --path_data=$DATA_DIR"

# Execute the command
$enqueue_compss_cmd

