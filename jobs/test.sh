#!/bin/bash
#SBATCH --account=project_2009050
#SBATCH --job-name=dmtl
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=./logs/out_%j_%x_%N.log  # includes time stamp (t), job ID(j), job name (x), and node name (N)
#SBATCH --error=./logs/err_%j_%x_%N.err

module --force purge
module load pytorch
source /projappl/project_2009050/code/fed/bin/activate
cd /projappl/project_2009050/fedai

export PYTHONPATH=$PYTHONPATH:/projappl/project_2009050/code/fedai/mytorch/lib/python3.11/site-packages
echo "Current PYTHONPATH: $PYTHONPATH"

srun python main.py n_rounds=3 algorithm=feddbe optimizer=sgd
