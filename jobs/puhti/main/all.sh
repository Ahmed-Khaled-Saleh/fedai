#!/bin/bash
#SBATCH --account=project_2009050
#SBATCH --job-name=fedai_all
#SBATCH --output=logs/fedai_%A_%a.out
#SBATCH --error=logs/fedai_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --array=0-304          # Number of algorithms (0 to N-1)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1             # Request 1 GPU per job
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


echo "Running task $SLURM_ARRAY_TASK_ID: Algorithm=$CURRENT_ALGO on Dataset=$CURRENT_DATA with Image Size=$IMG_SIZE"

# 3. Load your environment (Conda, modules, etc.)
# module load cuda
# source activate your_env
module --force purge
module load pytorch
source /projappl/project_2009050/fed/bin/activate
cd /projappl/project_2009050/fedai

export PYTHONPATH=$PYTHONPATH:/projappl/project_2009050/fed/lib/python3.12/site-packages
echo "Current PYTHONPATH: $PYTHONPATH"


# 4. Launch Hydra
# We override the 'algorithm' and 'data' groups specifically
python main.py \
    algorithm=$CURRENT_ALGO \
    data=$CURRENT_DATA \
    partitioner=pathological \
    model=lenet \
    model.name=$MODEL_NAME \
    model.img_size=$IMG_SIZE \
    $OPT_OVERRIDE \
    server=puhti \
    m=$CURRENT_M \
    num_clients=$CURRENT_NUM_CLIENTS \