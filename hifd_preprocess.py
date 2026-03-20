import os
import glob
import numpy as np
import scipy.io
import pandas as pd
import gc
from sklearn.preprocessing import StandardScaler

# ==========================================
# Configuration
# ==========================================
DATA_PATH = "hifd_dataset"
OUTPUT_DIR = "processed_tensors_hifd"
WINDOW_SIZE = 128
STEP_SIZE = 64

os.makedirs(OUTPUT_DIR, exist_ok=True)

import sys

def parse_hifd_file(file_path):
    try:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None, None, None
            
        mat_data = scipy.io.loadmat(file_path)
        
        # Accelerometer (mandatory)
        if 'ax' in mat_data and 'ay' in mat_data and 'az' in mat_data:
            acc = np.stack([
                mat_data['ax'].flatten(),
                mat_data['ay'].flatten(),
                mat_data['az'].flatten()
            ], axis=1)
        else:
            print(f"ERROR: Missing accelerometer keys in {file_path}")
            return None, None, None
            
        # Gyroscope
        if 'droll' in mat_data and 'dpitch' in mat_data and 'dyaw' in mat_data:
            gyro = np.stack([
                mat_data['droll'].flatten(),
                mat_data['dpitch'].flatten(),
                mat_data['dyaw'].flatten()
            ], axis=1)
        else:
            gyro = None
            
        # Heart Rate
        if 'heart' in mat_data:
            heart = mat_data['heart'].flatten()
        else:
            heart = None
            
        return acc, gyro, heart

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None, None, None

def create_hifd_dataset(mode):
    print(f"\n--- Generating HIFD Dataset for {mode} ---", flush=True)
    
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset path '{DATA_PATH}' does not exist!")
        sys.exit(1)

    X_imu_list, X_hr_list, y_list = [], [], []
    failure_count = 0
    MAX_LOG_FAILURES = 10
    
    try:
        subjects = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d)) and d.startswith('subject')]
    except Exception as e:
        print(f"ERROR: Failed to list subjects in {DATA_PATH}: {e}")
        sys.exit(1)

    subjects.sort()
    print(f"Found {len(subjects)} subjects: {subjects}", flush=True)
    
    if not subjects:
        print(f"ERROR: No subject directories found in {DATA_PATH}")
        sys.exit(1)

    category_map = {'fall': 1, 'non-fall': 0}
    
    for subject in subjects:
        for cat_dir, label in category_map.items():
            dir_path = os.path.join(DATA_PATH, subject, cat_dir)
            if not os.path.exists(dir_path):
                continue
            
            files = glob.glob(os.path.join(dir_path, "*.mat"))
            if not files:
                continue
                
            for f in files:
                acc, gyro, heart = parse_hifd_file(f)
                
                if acc is None:
                    failure_count += 1
                    if failure_count > MAX_LOG_FAILURES:
                        # Be silent after too many failures to keep log clean
                        pass
                    continue
                
                # Check if we have enough components for the mode
                if mode == '3axis':
                    imu_data = acc
                elif mode == '6axis':
                    if gyro is not None:
                        imu_data = np.hstack([acc, gyro])
                    else:
                        continue # Skip this file for 6-axis if no gyro
                else:
                    continue
                
                # Sliding Window
                for j in range(0, len(imu_data) - WINDOW_SIZE, STEP_SIZE):
                    window_imu = imu_data[j : j + WINDOW_SIZE]
                    
                    # For HR, handle missing HR (None)
                    if heart is not None:
                        window_heart = heart[j : j + WINDOW_SIZE]
                        avg_hr = np.mean(window_heart)
                    else:
                        avg_hr = 0.0 # Placeholder for missing HR
                    
                    X_imu_list.append(window_imu)
                    X_hr_list.append(avg_hr)
                    y_list.append(label)
                    
    if len(X_imu_list) == 0:
        print(f"ERROR: No windows generated for {mode}. Check constraints and file formats.")
        if failure_count > 0:
             print(f"Total failures encountered: {failure_count}")
        return # Skip saving if no data

    X_imu = np.array(X_imu_list, dtype=np.float32)
    X_hr = np.array(X_hr_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    print(f"Saving {mode}: IMU {X_imu.shape}, HR {X_hr.shape}, Labels {y.shape}")
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"Class distribution for {mode}: {dist}")
    
    if len(unique) < 2:
        print(f"WARNING: Only one class present in {mode} dataset. Experiments will be biased!")
    
    save_path = os.path.join(OUTPUT_DIR, f"data_hifd_{mode}.npz")
    np.savez_compressed(save_path, X_imu=X_imu, X_hr=X_hr, y=y)
    print(f"Successfully saved to {save_path}")


if __name__ == '__main__':
    print("HIFD Preprocessing Script Started", flush=True)
    create_hifd_dataset('3axis')
    create_hifd_dataset('6axis')
    print("Preprocessing HIFD Done.")




