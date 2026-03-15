#!/bin/bash
#SBATCH --account=project_462001088
#SBATCH --job-name=fedai_drichlet_cifar10
#SBATCH --output=logs/fedai_%A_%a.out
#SBATCH --error=logs/fedai_%A_%a.err
#SBATCH --partition=small-g
#SBATCH --array=0-119             # Number of algorithms (0 to N-1)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=36:00:00             # Adjust based on expected runtime

# 1. Define your array of algorithms (must match the names in your cs.store)
algos=(
    "fedavg_ft" "pfedme" "fedu" 
    "sfmtl" "fedala" "fedper" 
    "lgfedavg" "fedrep" "fedrod" 
    "fedbabu"
)

alpha=(0.1 0.3 0.5)  # Example alpha values for Dirichlet partitioning
m=(0.2 0.4 0.6 1.0)

combinations=()
for a in "${algos[@]}"; do
    for al in "${alpha[@]}"; do
        for mm in "${m[@]}"; do
            combinations+=("$a $al $mm")
        done
    done
done

# 3. Safety Check: Ensure the SLURM_ARRAY_TASK_ID is within bounds
if [ $SLURM_ARRAY_TASK_ID -ge ${#combinations[@]} ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID out of bounds. Max index is $((${#combinations[@]} - 1))"
    exit 1
fi

# 4. Extract parameters for this specific task
CURRENT_COMBO=${combinations[$SLURM_ARRAY_TASK_ID]}
read -r CURRENT_ALGO CURRENT_ALPHA CURRENT_M <<< "$CURRENT_COMBO"

echo "------------------------------------------------"
echo "Job ID: $SLURM_ARRAY_JOB_ID | Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running: $CURRENT_ALGO | Alpha: $CURRENT_ALPHA | M: $CURRENT_M"
echo "------------------------------------------------"


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

IMG_SIZE="[3,32,32]"
echo "Running task $SLURM_ARRAY_TASK_ID: Algorithm=$CURRENT_ALGO on Dataset=cifar10"

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
    data=cifar10 \
    partitioner=dirichlet \
    partitioner.alpha=$CURRENT_ALPHA \
    m=$CURRENT_M \
    model=lenet \
    model.name=lenet_cifar10 \
    model.img_size=$IMG_SIZE \
    $OPT_OVERRIDE \
    server=lumi