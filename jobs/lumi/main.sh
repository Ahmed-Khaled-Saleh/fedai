#!/bin/bash
#SBATCH --account=project_462001088
#SBATCH --job-name=fedai_drichlet_cifar10
#SBATCH --output=logs/fedai_%A_%a.out
#SBATCH --error=logs/fedai_%A_%a.err
#SBATCH --partition=small-g
#SBATCH --array=0-303             # Number of algorithms (0 to N-1)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=36:00:00             # Adjust based on expected runtime

# 1. Define your array of algorithms (must match the names in your cs.store)
algos=(
    "local" "fedavg" "fedavg_ft" "pfedme" "fedu" 
    "sfmtl" "ditto" "fedprox" "apfl" "fedala" 
    "ifca" "fedper" "lgfedavg" "fedrep" "fedrod" 
    "fedbabu" "gpfl" "feddbe" "fedas"
)

datasets=(
    "cifar10" "fashionmnist" "cinic10" "mnist_rotated_batched"
)
m=(1.0 0.3)
num_clients=(20 100)

combinations=()
for a in "${algos[@]}"; do
    for d in "${datasets[@]}"; do
        for mm in "${m[@]}"; do
            for nc in "${num_clients[@]}"; do
                combinations+=("$a|$d|$mm|$nc") # Use a separator
            done
        done
    done
done

# Get the pair for THIS specific task index
current_pair=${combinations[$SLURM_ARRAY_TASK_ID]}

# Split the pair back into two variables
CURRENT_ALGO=$(echo $current_pair | cut -d'|' -f1)
CURRENT_DATA=$(echo $current_pair | cut -d'|' -f2)
CURRENT_M=$(echo $current_pair | cut -d'|' -f3)
CURRENT_NUM_CLIENTS=$(echo $current_pair | cut -d'|' -f4)


OPT_OVERRIDE=""

# Logic to switch optimizer for specific algorithms
if [ "$CURRENT_ALGO" == "pfedme" ]; then
    OPT_OVERRIDE="optimizer=pfedme"
elif [ "$CURRENT_ALGO" == "fedprox" ]; then
    OPT_OVERRIDE="optimizer=fedprox"
elif [ "$CURRENT_ALGO" == "perfedavg" ]; then
    OPT_OVERRIDE="optimizer=perfedavg"
else
    # Default optimizer for everyone else (e.g., sgd)
    OPT_OVERRIDE="optimizer=sgd"
fi

IMG_SIZE=""
if [ "$CURRENT_DATA" == "mnist_rotated_batched" ]; then
    IMG_SIZE="[1,28,28]"
elif [ "$CURRENT_DATA" == "fashionmnist" ]; then
    IMG_SIZE="[1,28,28]"
elif [ "$CURRENT_DATA" == "cinic10" ]; then
    IMG_SIZE="[3,32,32]"
else
    IMG_SIZE="[3,32,32]"
fi 

MODEL_NAME=""
if [ "$CURRENT_DATA" == "mnist_rotated_batched" ]; then
    MODEL_NAME="lenet_fedavg"
elif [ "$CURRENT_DATA" == "fashionmnist" ]; then
    MODEL_NAME="lenet_fedavg"
elif [ "$CURRENT_DATA" == "cinic10" ]; then
    MODEL_NAME="lenet_cifar10"
else
    MODEL_NAME="lenet_cifar10"
fi 

if [ "$CURRENT_DATA" == "mnist_rotated_batched" ]; then
    CURRENT_PARTITIONER="rotated"
else
    CURRENT_PARTITIONER="pathological"
fi



echo "Running task $SLURM_ARRAY_TASK_ID: Algorithm=$CURRENT_ALGO on Dataset=$CURRENT_DATA with Image Size=$IMG_SIZE"


SIF_FILE="/scratch/project_462001088/EasyBuild/SW/container/PyTorch/2.6.0-rocm-6.2.4-python-3.12-Mycontainer-singularity-20250410/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0-dockerhash-ef203c810cc9.sif"  # Update this path to your actual container
cd /projappl/project_462001088/fedai

module purge
module load LUMI/24.03
module load PyTorch/2.6.0-rocm-6.2.4-python-3.12-Mycontainer-singularity-20250410

singularity exec \
    -B /dev/dri \
    -B /dev/kfd \
    --pwd /projappl/project_462001088/fedai \
    -B /projappl/project_462001088 \
    -B /scratch/project_462001088 \
    $SIF_FILE \
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