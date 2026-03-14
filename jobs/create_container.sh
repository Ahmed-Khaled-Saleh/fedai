#!/bin/bash -l
#SBATCH --job-name=setup   # Job name
#SBATCH --output=setup.o%j # Name of stdout output file
#SBATCH --error=setup.e%j  # Name of stderr error file
#SBATCH --partition=small       # Partition name
#SBATCH --ntasks=1              # One task (process)
#SBATCH --mem=224G              # Memory request
#SBATCH --time=02:00:00         # Run time (hh:mm:ss)
#SBATCH --account=project_462001088  # Project for billing



installdir="/scratch/project_462001088/$USER/DEMO1"
# mkdir -p "$installdir/tmp" ; cd "$installdir/tmp"
# eb --copy-ec PyTorch-2.6.0-rocm-6.2.4-python-3.12-singularity-20250410.eb PyTorch-2.6.0-rocm-6.2.4-python-3.12-Mycontainer-singularity-20250410.eb
# sed -e "s|^\(versionsuffix.*\)-singularity-\(.*\)|\1-Mycontainer-singularity-\2|" -i PyTorch-2.6.0-rocm-6.2.4-python-3.12-Mycontainer-singularity-20250410.eb
# eb PyTorch-2.6.0-rocm-6.2.4-python-3.12-Mycontainer-singularity-20250410.eb




# cat > lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0-Mycontainer.def <<EOF

# Bootstrap: localimage

# From: /scratch/project_462001088/EasyBuild/SW/container/PyTorch/2.6.0-rocm-6.2.4-python-3.12-Mycontainer-singularity-20250410/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0-dockerhash-ef203c810cc9.sif

# %post

# zypper -n install -y Mesa libglvnd libgthread-2_0-0 hostname

# EOF

cd "$installdir/tmp"

module purge
module load LUMI/24.03
# module load PyTorch/2.7.1-rocm-6.2.4-python-3.12-Mycontainer-singularity-20250827

# 1. Capture the value of $SIF into a local variable so it persists
# after the module is unloaded.
# 1. Fix the typo here
# CONTAINERFILE="$SIF"
# echo "SIF path: $SIF"
# 2. Now unload the module safely
# module unload PyTorch/2.7.1-rocm-6.2.4-python-3.12-Mycontainer-singularity-20250827

# # 3. Load the build tools
# module load systools/24.03
# --- Use local proot binary instead of systools/24.03 ---
PROOT_BIN="/scratch/project_462001088/"   # <-- adjust this path to where your proot binary lives
export PATH="$PROOT_BIN:$PATH"

# Verify proot is found before proceeding
if ! command -v proot &>/dev/null; then
    echo "ERROR: proot not found in $PROOT_BIN" >&2
    exit 1
fi
echo "Using proot: $(which proot)"

# Build the container using proot-based unprivileged build
singularity build --force \
    /scratch/project_462001088/EasyBuild/SW/container/PyTorch/2.6.0-rocm-6.2.4-python-3.12-Mycontainer-singularity-20250410/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0-dockerhash-ef203c810cc9.sif \
    lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.6.0-Mycontainer.def
    
