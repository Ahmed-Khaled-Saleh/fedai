#!/bin/bash
#SBATCH --job-name=fedai_mnist
#SBATCH --output=logs/fedai_%A_%a.out
#SBATCH --error=logs/fedai_%A_%a.err
#SBATCH --array=0-17                # Number of algorithms (0 to N-1)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1             # Request 1 GPU per job
#SBATCH --time=8:00:00             # Adjust based on expected runtime

# 1. Define your array of algorithms (must match the names in your cs.store)
algos=(
    "fedavg" "fedavg_ft" "pfedme" "fedu" "sfmtl" 
    "perfedavg" "ditto" "fedprox" "apfl" "fedala" 
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
else
    # Default optimizer for everyone else (e.g., sgd)
    OPT_OVERRIDE="optimizer=sgd"
fi


echo "Running task $SLURM_ARRAY_TASK_ID: Algorithm=$CURRENT_ALGO on Dataset=mnist_rotated_batched"

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
    data=mnist_rotated_batched \
    server=puhti \
    $OPT_OVERRIDE