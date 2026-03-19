import subprocess
import time
import os

def run_command(cmd, log_file):
    print(f"Running: {cmd}")
    with open(log_file, "w") as f:
        process = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        return process

def main():
    # 1. Individual Run
    print("Step 1: Running Individual CNN-GRU experiments...")
    p1 = run_command("python run_cnn_gru_individual.py", "cnn_gru_individual.log")
    
    # 2. Hyperparameter Tuning (9axis)
    print("Step 2: Starting Hyperparameter Tuning for CNN-GRU (Unidirectional)...")
    p2 = run_command("python tune_cnn_gru.py --dataset fall_detection --mode 9axis", "cnn_gru_tuning_fd.log")
    p3 = run_command("python tune_cnn_gru.py --dataset umafall --mode 9axis", "cnn_gru_tuning_umafall.log")
    
    print("Step 2b: Starting Hyperparameter Tuning for CNN-BiGRU (Bidirectional)...")
    p2b = run_command("python tune_cnn_gru.py --dataset fall_detection --mode 9axis --bidirectional", "cnn_bigru_tuning_fd.log")
    p3b = run_command("python tune_cnn_gru.py --dataset umafall --mode 9axis --bidirectional", "cnn_bigru_tuning_umafall.log")
    
    # 3. Full Comparison
    print("Step 3: Running Full Model Comparison (including CNN-GRU)...")
    p4 = run_command("python dl_experiments.py", "dl_experiments_full.log")
    p5 = run_command("python dl_experiments_umafall.py", "dl_experiments_umafall_full.log")
    
    print("\nAll experiment processes have been started in the background.")
    print("Results and logs will be saved as they progress.")
    print("Logs: cnn_gru_individual.log, cnn_gru_tuning_fd.log, cnn_gru_tuning_umafall.log, dl_experiments_full.log, dl_experiments_umafall_full.log")

if __name__ == "__main__":
    main()
