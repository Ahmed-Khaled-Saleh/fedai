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

for algo in algorithms:
    # Determine optimizer mapping
    opt = "pfedme" if algo == "pfedme" else ("fedprox" if algo == "fedprox" else "sgd")
    
    # Construct the sbatch command using your specific requirements
    cmd = [
        "sbatch",
        "--account=project_2009050",
        "--job-name=fedai",
        "--partition=gpu",
        "--ntasks=1",
        "--cpus-per-task=4",
        "--mem=100G",
        "--time=08:00:00",
        "--gres=gpu:v100:1",
        "--output=./logs/out_%j_%x_%N.log",
        "--error=./logs/err_%j_%x_%N.err",
        "--wrap", f"python main.py algorithm={algo} optimizer={opt} data=mnist_rotated_batched server=puhti"
    ]
    
    # Submit the job
    print(f"Submitting {algo}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"  Success: {result.stdout.strip()}")
    else:
        print(f"  Failed: {result.stderr.strip()}")
    
    # Small delay to keep the scheduler happy
    time.sleep(0.5)

print("Submission complete. Use 'squeue -u $USER' to monitor your jobs.")