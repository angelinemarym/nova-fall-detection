import subprocess
import os
import time

def run_command(cmd, log_file):
    print(f"Running: {cmd}")
    with open(log_file, "w") as f:
        process = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        return process

def main():
    # 1. Preprocessing (assuming it's not already running or to be sure)
    print("Step 1: Preprocessing HIFD Dataset...")
    p1 = subprocess.run("python hifd_preprocess.py", shell=True)
    
    # 2. ML Experiments
    print("Step 2: Running ML Experiments on HIFD...")
    p2 = run_command("python ml_experiments_hifd.py", "ml_hifd.log")
    
    # 3. DL Experiments
    print("Step 3: Running DL Experiments on HIFD...")
    p3 = run_command("python dl_experiments_hifd.py", "dl_hifd.log")
    
    print("\nExperiments are running in the background.")
    print("Check ml_hifd.log and dl_hifd.log for progress.")

if __name__ == "__main__":
    main()
