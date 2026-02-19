import os
import glob
import numpy as np
import pandas as pd
import gc

# ==========================================
# Configuration
# ==========================================
DATA_PATH = "fall_detection_data/Dataset"
OUTPUT_DIR = "processed_tensors"
WINDOW_SIZE = 128
STEP_SIZE = 64

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_file_optimized(file_path, mode):
    try:
        # Load Raw Data
        raw_df = pd.read_csv(file_path, header=None, skiprows=1)
        df = raw_df.iloc[:, :6].copy()
        df.columns = ['t', 'x', 'y', 'z', 'a', 'sensor']

        # Cleanup numeric strings
        for col in ['x', 'y', 'z']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Cleanup sensor names (lowercase)
        df['sensor'] = df['sensor'].astype(str).str.strip().str.lower()

        # 1. Extract Heart Rate (Average for the whole file)
        # Assuming HR is either tagged 'hrt' in column 3 or 'heart'/'hrt' in sensor column
        hrt_mask = (raw_df.iloc[:, 3].astype(str).str.strip().str.lower() == 'hrt') | (df['sensor'] == 'hrt') | (df['sensor'] == 'heart')
        hrt_rows = raw_df[hrt_mask]
        
        file_bpm = 75.0
        if not hrt_rows.empty:
            # HR values are typically in column 1 or 2. Take the mean.
            for val_col in [1, 2]:
                bpm_vals = pd.to_numeric(hrt_rows.iloc[:, val_col], errors='coerce').dropna()
                if not bpm_vals.empty:
                    file_bpm = bpm_vals.mean()
                    break
                    
        if np.isnan(file_bpm): file_bpm = 75.0

        # 2. Extract Accelerometer
        acc_df = df[df['sensor'] == 'acc'].dropna(subset=['x', 'y', 'z']).reset_index(drop=True)
        
        if acc_df.empty: 
            return None, None
            
        acc_vals = acc_df[['x', 'y', 'z']].values
        
        # 3. Align by Sequence Index (Shortest Length Truncation)
        # Because 't' is identical (2.92E+12) for all rows, we align by chronological sequence.
        final_data = acc_vals
        min_len = len(acc_vals)
        
        if mode in ['6axis', '9axis']:
            gyro_df = df[df['sensor'] == 'gyro'].dropna(subset=['x', 'y', 'z']).reset_index(drop=True)
            if not gyro_df.empty:
                min_len = min(min_len, len(gyro_df))
                gyro_vals = gyro_df.loc[:min_len-1, ['x', 'y', 'z']].values
            else:
                gyro_vals = np.zeros((min_len, 3)) # Dummy zeros if missing
                
            final_data = final_data[:min_len]
            final_data = np.hstack([final_data, gyro_vals])

        if mode == '9axis':
            mag_df = df[df['sensor'] == 'mgm'].dropna(subset=['x', 'y', 'z']).reset_index(drop=True)
            if not mag_df.empty:
                min_len = min(min_len, len(mag_df))
                mag_vals = mag_df.loc[:min_len-1, ['x', 'y', 'z']].values
            else:
                mag_vals = np.zeros((min_len, 3)) # Dummy zeros if missing
                
            final_data = final_data[:min_len]
            final_data = np.hstack([final_data, mag_vals])

        return final_data, file_bpm

    except Exception as e:
        return None, None

def create_dataset(mode):
    print(f"--- Generating Dataset for {mode} ---")
    X_imu_list, X_hr_list, y_list = [], [], []
    
    categories = {'adl': 0, 'fall': 1}
    
    for cat_name, label in categories.items():
        search_path = os.path.join(DATA_PATH, cat_name, "**", "*.csv")
        files = glob.glob(search_path, recursive=True)
        print(f"Processing {len(files)} files for class '{cat_name}'...")
        
        for i, f in enumerate(files):
            if i > 0 and i % 100 == 0: gc.collect()
            
            res = parse_file_optimized(f, mode)
            if res[0] is None: continue
            
            imu_vals, bpm = res
            
            if len(imu_vals) < WINDOW_SIZE:
                continue

            # Sliding Window
            for j in range(0, len(imu_vals) - WINDOW_SIZE, STEP_SIZE):
                window_imu = imu_vals[j : j + WINDOW_SIZE]
                
                X_imu_list.append(window_imu)
                X_hr_list.append(bpm)
                y_list.append(label)

    # Convert to NumPy
    if len(X_imu_list) == 0:
        print(f"ERROR: No data generated for {mode}.")
        return

    X_imu = np.array(X_imu_list, dtype=np.float32)
    X_hr = np.array(X_hr_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    print(f"Saving {mode}: IMU {X_imu.shape}, HR {X_hr.shape}, Labels {y.shape}")
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, f"data_{mode}.npz"),
        X_imu=X_imu, X_hr=X_hr, y=y
    )
    
    del X_imu, X_hr, y, X_imu_list, X_hr_list, y_list
    gc.collect()

if __name__ == '__main__':
    create_dataset('3axis')
    create_dataset('6axis')
    create_dataset('9axis')
    print("Done.")