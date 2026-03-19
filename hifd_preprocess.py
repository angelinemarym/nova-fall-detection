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
            
        data = scipy.io.loadmat(file_path)
        data_keys = [k for k in data.keys() if not k.startswith('__')]
        if not data_keys:
            print(f"No data keys found in {file_path}. Keys: {list(data.keys())}")
            return None, None, None
            
        data_key = data_keys[0]
        matrix = data[data_key]
        
        if len(matrix.shape) == 2:
            # Case A: 11-12 columns as per README (Acc, Quat, Gyro, HR)
            if matrix.shape[1] >= 11:
                acc = matrix[:, 0:3]
                gyro = matrix[:, 7:10] 
                heart = matrix[:, 10] 
                return acc, gyro, heart
                
            # Case B: 6 columns (Likely Acc + Gyro, no HR or Quat)
            elif matrix.shape[1] == 6:
                acc = matrix[:, 0:3]
                gyro = matrix[:, 3:6]
                return acc, gyro, None
                
            # Case C: 3 columns (Just Acc)
            elif matrix.shape[1] == 3:
                acc = matrix[:, 0:3]
                return acc, None, None

        # Case D: Separate keys (ax, ay, az, droll, dpitch, dyaw, heart, etc.)
        if all(k in data for k in ['ax', 'ay', 'az']):
            acc = np.hstack([data['ax'].reshape(-1, 1), data['ay'].reshape(-1, 1), data['az'].reshape(-1, 1)])
            gyro = None
            if all(k in data for k in ['droll', 'dpitch', 'dyaw']):
                 gyro = np.hstack([data['droll'].reshape(-1, 1), data['dpitch'].reshape(-1, 1), data['dyaw'].reshape(-1, 1)])
            
            heart = data.get('heart')
            if heart is not None:
                heart = heart.flatten()
            
            return acc, gyro, heart

        print(f"DEBUG: File {file_path} - Found '{data_key}' with shape {matrix.shape}. Keys: {data_keys}")
        for k in data_keys:
            if hasattr(data[k], 'shape'):
                print(f"  - Key '{k}' shape: {data[k].shape}")
        print(f"ERROR: Unsupported structure in {file_path} (Shape {matrix.shape})")
        return None, None, None


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




