#!/bin/bash
#SBATCH --account=project_2009050
#SBATCH --job-name=fedai_fashionmnist
#SBATCH --output=logs/fedai_%A_%a.out
#SBATCH --error=logs/fedai_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --array=0-16               # Number of algorithms (0 to N-1)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1             # Request 1 GPU per job
#SBATCH --time=24:00:00             # Adjust based on expected runtime

# 1. Define your array of algorithms (must match the names in your cs.store)
algos=(
    "fedavg" "fedavg_ft" "pfedme" "fedu" "sfmtl" 
    "ditto" "fedprox" "apfl" "fedala" 
    "ifca" "fedper" "lgfedavg" "fedrep" "fedrod" 
    "fedbabu" "gpfl" "feddbe"
)

# 2. Get the specific algorithm for THIS task
CURRENT_ALGO=${algos[$SLURM_ARRAY_TASK_ID]}

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

IMG_SIZE="[1,28,28]"
echo "Running task $SLURM_ARRAY_TASK_ID: Algorithm=$CURRENT_ALGO on Dataset=fashionmnist"

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
    data=fashionmnist \
    model=lenet \
    model.name=lenet_fedavg \
    model.img_size=$IMG_SIZE \
    server=puhti \
    $OPT_OVERRIDE