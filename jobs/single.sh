#!/bin/bash
#SBATCH --account=project_2009050
#SBATCH --job-name=fedai_mnist_rotated_batched
#SBATCH --output=logs/fedai_%A_%a.out
#SBATCH --error=logs/fedai_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1             # Request 1 GPU per job
#SBATCH --time=8:00:00             # Adjust based on expected runtime


module --force purge
module load pytorch
source /projappl/project_2009050/fed/bin/activate
cd /projappl/project_2009050/fedai

export PYTHONPATH=$PYTHONPATH:/projappl/project_2009050/fed/lib/python3.12/site-packages
echo "Current PYTHONPATH: $PYTHONPATH"

IMG_SIZE="[3,32,32]"

python main.py \
    algorithm=fedas \
    data=cinic10 \
    model=lenet \
    model.name=lenet_cifar10 \
    model.img_size=$IMG_SIZE \
    server=puhti \
    optimizer=sgd \