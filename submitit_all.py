import subprocess
import os
import time

# List of algorithms to run
algorithms = ["fedavg", "fedavg_ft", "pfedme", "fedu", "sfmtl", 
              "perfedavg", "ditto", "fedprox", "apfl", "fedala", 
              "ifca", "fedper", "lgfedavg", "fedrep", "fedrod", 
              "fedbabu", "gpfl", "feddbe"]

# Ensure logs directory exists
os.makedirs("./logs", exist_ok=True)
SBATCH_PATH = "/usr/bin/sbatch"
for algo in algorithms:
    # Determine optimizer mapping
    opt = "pfedme" if algo == "pfedme" else ("fedprox" if algo == "fedprox" else "sgd")
    
    # Construct the sbatch command using your specific requirements
    cmd = (
        f"{SBATCH_PATH} --account=project_2009050 --job-name=fedai --partition=gpu "
        f"--ntasks=1 --cpus-per-task=4 --mem=100G --time=08:00:00 "
        f"--gres=gpu:v100:1 --output=./logs/out_%j_%x_%N.log "
        f"--error=./logs/err_%j_%x_%N.err "
        f"--wrap=\"python main.py algorithm={algo} optimizer={opt} data=mnist_rotated_batched server=puhti\""
    )
    
    # Submit the job
    print(f"Submitting {algo}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
#     result = subprocess.run(
#     " ".join(cmd), 
#     shell=True, 
#     executable='/bin/bash', 
#     capture_output=True, 
#     text=True
# )
    
    if result.returncode == 0:
        print(f"  Success: {result.stdout.strip()}")
    else:
        print(f"  Failed: {result.stderr.strip()}")
        print(f"  STDERR: {result.stderr.strip()}")
    
    # Small delay to keep the scheduler happy
    time.sleep(0.5)

print("Submission complete. Use 'squeue -u $USER' to monitor your jobs.")