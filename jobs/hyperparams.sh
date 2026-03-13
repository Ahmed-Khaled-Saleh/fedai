#!/bin/bash
#SBATCH --account=project_2009050
#SBATCH --job-name=fedala_tuning
#SBATCH --output=logs/fedai_%A_%a.out
#SBATCH --error=logs/fedai_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=8:00:00
#SBATCH --array=0-11            # <--- Update this to (Total Combos - 1)

# 1. Define your hyperparameter grids
algos=("fedala")
layer_idxs=(0 1 2 3)
rand_percents=("0.2" "0.5" "0.8")

# 2. Build the combinations list
combinations=()
for a in "${algos[@]}"; do
    for l in "${layer_idxs[@]}"; do
        for r in "${rand_percents[@]}"; do
            combinations+=("$a $l $r")
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
read -r ALGO L_IDX R_PERC <<< "$CURRENT_COMBO"

echo "------------------------------------------------"
echo "Job ID: $SLURM_ARRAY_JOB_ID | Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running: $ALGO | Layer: $L_IDX | Rand%: $R_PERC"
echo "------------------------------------------------"

# ... (Module loading and environment setup here) ...


module --force purge
module load pytorch
source /projappl/project_2009050/fed/bin/activate
cd /projappl/project_2009050/fedai

export PYTHONPATH=$PYTHONPATH:/projappl/project_2009050/fed/lib/python3.12/site-packages
echo "Current PYTHONPATH: $PYTHONPATH"


# 5. Run the training
python main.py \
    algorithm=$ALGO \
    data=cinic10 \
    model=lenet \
    model.name=lenet_cifar10 \
    model.img_size="[3,32,32]" \
    server=puhti \
    optimizer=sgd \
    algorithm.layer_idx=$L_IDX \
    algorithm.rand_percent=$R_PERC