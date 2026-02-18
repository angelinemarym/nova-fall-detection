import os
import glob
import numpy as np
import pandas as pd
import gc

# ==========================================
# Configuration (No HR)
# ==========================================
DATA_PATH = "fall_detection_data/Dataset_no_heart"
OUTPUT_DIR = "processed_tensors_no_hr"
WINDOW_SIZE = 128
STEP_SIZE = 64

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_file_optimized(file_path, mode):
    try:
        raw_df = pd.read_csv(file_path, header=None, skiprows=1)
        df = raw_df.iloc[:, :6].copy()
        df.columns = ['t', 'x', 'y', 'z', 'a', 'sensor']

        for col in ['x', 'y', 'z']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['sensor'] = df['sensor'].astype(str).str.strip().str.lower()

        # 1. Extract Accelerometer
        acc_df = df[df['sensor'] == 'acc'].dropna(subset=['x', 'y', 'z']).reset_index(drop=True)
        if acc_df.empty: return None
            
        acc_vals = acc_df[['x', 'y', 'z']].values
        final_data = acc_vals
        min_len = len(acc_vals)
        
        # 2. Extract Gyroscope
        if mode in ['6axis', '9axis']:
            gyro_df = df[df['sensor'] == 'gyro'].dropna(subset=['x', 'y', 'z']).reset_index(drop=True)
            if not gyro_df.empty:
                min_len = min(min_len, len(gyro_df))
                gyro_vals = gyro_df.loc[:min_len-1, ['x', 'y', 'z']].values
            else:
                gyro_vals = np.zeros((min_len, 3))
                
            final_data = final_data[:min_len]
            final_data = np.hstack([final_data, gyro_vals])

        # 3. Extract Magnetometer
        if mode == '9axis':
            mag_df = df[df['sensor'] == 'mag'].dropna(subset=['x', 'y', 'z']).reset_index(drop=True)
            if not mag_df.empty:
                min_len = min(min_len, len(mag_df))
                mag_vals = mag_df.loc[:min_len-1, ['x', 'y', 'z']].values
            else:
                mag_vals = np.zeros((min_len, 3))
                
            final_data = final_data[:min_len]
            final_data = np.hstack([final_data, mag_vals])

        return final_data

    except Exception as e:
        return None

def create_dataset(mode):
    print(f"--- Generating No-HR Dataset for {mode} ---")
    X_imu_list, y_list = [], []
    categories = {'adl': 0, 'fall': 1}
    
    for cat_name, label in categories.items():
        search_path = os.path.join(DATA_PATH, cat_name, "**", "*.csv")
        files = glob.glob(search_path, recursive=True)
        
        for i, f in enumerate(files):
            if i > 0 and i % 100 == 0: gc.collect()
            
            imu_vals = parse_file_optimized(f, mode)
            if imu_vals is None or len(imu_vals) < WINDOW_SIZE:
                continue

            # Sliding Window
            for j in range(0, len(imu_vals) - WINDOW_SIZE, STEP_SIZE):
                window_imu = imu_vals[j : j + WINDOW_SIZE]
                X_imu_list.append(window_imu)
                y_list.append(label)

    if len(X_imu_list) == 0:
        print(f"ERROR: No data generated for {mode}.")
        return

    X_imu = np.array(X_imu_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    print(f"Saving {mode}: IMU {X_imu.shape}, Labels {y.shape}")
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, f"data_{mode}_no_hr.npz"),
        X_imu=X_imu, y=y
    )
    del X_imu, y, X_imu_list, y_list
    gc.collect()

if __name__ == '__main__':
    create_dataset('3axis')
    create_dataset('6axis')
    create_dataset('9axis')