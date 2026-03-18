#!/bin/bash
#SBATCH --account=project_462001088
#SBATCH --job-name=fedai_standardg
#SBATCH --output=logs/fedai_%A_%a.out
#SBATCH --error=logs/fedai_%A_%a.err
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=36:00:00
#SBATCH --array=0-2   # 3 nodes → 24 experiments total

# ================================
# CONFIG
# ================================
TASKS_PER_NODE=8
GLOBAL_OFFSET=$((SLURM_ARRAY_TASK_ID * TASKS_PER_NODE))

SIF_FILE="/scratch/project_462001088/EasyBuild/SW/container/PyTorch/2.6.0-rocm-6.2.4-python-3.12-Mycontainer-singularity-20250410/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0-dockerhash-ef203c810cc9.sif"

WORKDIR="/projappl/project_462001088/fedai"
COMB_FILE="$WORKDIR/combos.txt"

cd $WORKDIR

# ================================
# LOAD MODULES
# ================================
module purge
module load LUMI/24.03
module load PyTorch/2.6.0-rocm-6.2.4-python-3.12-Mycontainer-singularity-20250410

# ================================
# BUILD COMBINATIONS (ONLY ONCE)
# ================================
if [ ! -f "$COMB_FILE" ]; then
    echo "Generating combinations file..."

    algos=(
        "local" "fedavg" "fedavg_ft" "pfedme" "fedu" 
        "sfmtl" "ditto" "fedprox" "apfl" "fedala" 
        "ifca" "fedper" "lgfedavg" "fedrep" "fedrod" 
        "fedbabu" "gpfl" "feddbe" "fedas"
    )

    datasets=(
        "cifar10" "fashionmnist" "cinic10" "mnist_rotated_batched"
    )

    num_clients=(20 100)

    > $COMB_FILE

    for a in "${algos[@]}"; do
        for d in "${datasets[@]}"; do
            for nc in "${num_clients[@]}"; do
                echo "$a|$d|$nc" >> $COMB_FILE
            done
        done
    done
fi

# ================================
# PARALLEL EXECUTION (8 per node)
# ================================
srun --ntasks=8 --cpu-bind=cores bash -c '
LOCAL_ID=$SLURM_PROCID
GLOBAL_ID=$((GLOBAL_OFFSET + LOCAL_ID))

# Get this experiment
current_pair=$(sed -n "$((GLOBAL_ID+1))p" '"$COMB_FILE"')

if [ -z "$current_pair" ]; then
    echo "No more experiments for GLOBAL_ID=$GLOBAL_ID"
    exit 0
fi

CURRENT_ALGO=$(echo $current_pair | cut -d"|" -f1)
CURRENT_DATA=$(echo $current_pair | cut -d"|" -f2)
CURRENT_NUM_CLIENTS=$(echo $current_pair | cut -d"|" -f3)

# ================================
# DERIVED PARAMETERS
# ================================
CURRENT_M=0.5
if [ "$CURRENT_NUM_CLIENTS" -eq 20 ]; then
    CURRENT_M=1.0
elif [ "$CURRENT_NUM_CLIENTS" -eq 100 ]; then
    CURRENT_M=0.5
fi

# Optimizer
if [ "$CURRENT_ALGO" == "pfedme" ]; then
    OPT_OVERRIDE="optimizer=pfedme"
elif [ "$CURRENT_ALGO" == "fedprox" ]; then
    OPT_OVERRIDE="optimizer=fedprox"
else
    OPT_OVERRIDE="optimizer=sgd"
fi

# Image size
if [ "$CURRENT_DATA" == "mnist_rotated_batched" ] || [ "$CURRENT_DATA" == "fashionmnist" ]; then
    IMG_SIZE="[1,28,28]"
else
    IMG_SIZE="[3,32,32]"
fi

# Model name
if [ "$CURRENT_DATA" == "mnist_rotated_batched" ] || [ "$CURRENT_DATA" == "fashionmnist" ]; then
    MODEL_NAME="lenet_fedavg"
else
    MODEL_NAME="lenet_cifar10"
fi

# Partitioner
if [ "$CURRENT_DATA" == "mnist_rotated_batched" ]; then
    CURRENT_PARTITIONER="rotated"
else
    CURRENT_PARTITIONER="pathological"
fi

# ================================
# GPU BINDING (CRITICAL)
# ================================
export ROCR_VISIBLE_DEVICES=$LOCAL_ID

echo "======================================"
echo "Node: $SLURM_ARRAY_TASK_ID | GPU: $LOCAL_ID | Global: $GLOBAL_ID"
echo "Algo: $CURRENT_ALGO | Data: $CURRENT_DATA | Clients: $CURRENT_NUM_CLIENTS"
echo "======================================"

# ================================
# RUN
# ================================
singularity exec \
    -B /dev/dri \
    -B /dev/kfd \
    --pwd '"$WORKDIR"' \
    -B /projappl/project_462001088 \
    -B /scratch/project_462001088 \
    '"$SIF_FILE"' \
    python main.py \
        algorithm=$CURRENT_ALGO \
        data=$CURRENT_DATA \
        partitioner=$CURRENT_PARTITIONER \
        model=lenet \
        model.name=$MODEL_NAME \
        model.img_size=$IMG_SIZE \
        $OPT_OVERRIDE \
        server=lumi \
        m=$CURRENT_M \
        num_clients=$CURRENT_NUM_CLIENTS
'