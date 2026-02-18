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
BATCH_SIZE = 100  # Process 100 files, then save and clear RAM

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_file(file_path, mode):
    """Parses one CSV file. Returns fused dataframe and heart rate data."""
    try:
        # Read Raw
        raw_df = pd.read_csv(file_path, header=None, skiprows=1)
        df = raw_df.iloc[:, :6].copy()
        df.columns = ['t', 'x', 'y', 'z', 'a', 'sensor']
        
        # Coerce types
        for col in ['t', 'x', 'y', 'z']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Extract HR
        hrt_rows = raw_df[raw_df.iloc[:, 3].astype(str).str.strip() == 'hrt']
        hrt_data = pd.DataFrame({
            't': pd.to_numeric(hrt_rows.iloc[:, 0], errors='coerce'),
            'bpm': pd.to_numeric(hrt_rows.iloc[:, 1], errors='coerce')
        }).dropna()

        # Extract Sensors
        acc_df = df[df['sensor'].astype(str).str.strip() == 'acc'].dropna(subset=['t', 'x', 'y', 'z']).copy()
        acc_df = acc_df[['t', 'x', 'y', 'z']].rename(columns={'x': 'acc_x', 'y': 'acc_y', 'z': 'acc_z'})
        
        gyro_df = pd.DataFrame()
        mag_df = pd.DataFrame()

        if mode in ['6axis', '9axis']:
            gyro_df = df[df['sensor'].astype(str).str.strip() == 'gyro'].dropna(subset=['t', 'x', 'y', 'z']).copy()
            gyro_df = gyro_df[['t', 'x', 'y', 'z']].rename(columns={'x': 'gyro_x', 'y': 'gyro_y', 'z': 'gyro_z'})
        
        if mode == '9axis':
            mag_df = df[df['sensor'].astype(str).str.strip() == 'mag'].dropna(subset=['t', 'x', 'y', 'z']).copy()
            mag_df = mag_df[['t', 'x', 'y', 'z']].rename(columns={'x': 'mag_x', 'y': 'mag_y', 'z': 'mag_z'})

        # Merge Logic
        if mode == '3axis':
            fused_df = acc_df
        elif mode == '6axis':
            fused_df = pd.merge(acc_df, gyro_df, on='t', how='outer')
        elif mode == '9axis':
            fused_df = pd.merge(acc_df, gyro_df, on='t', how='outer')
            fused_df = pd.merge(fused_df, mag_df, on='t', how='outer')

        # Interpolate
        fused_df = fused_df.sort_values('t').interpolate(method='linear', limit_direction='both').dropna()

        # Normalize Time
        if not fused_df.empty:
            min_t = fused_df['t'].min()
            fused_df['t'] = (fused_df['t'] - min_t) / 1e6
            if not hrt_data.empty:
                hrt_data['t'] = (hrt_data['t'] - min_t) / 1e6

        return fused_df, hrt_data
    except Exception as e:
        return None, None

def process_and_save(mode):
    print(f"\n--- Processing {mode} ---")
    
    # Identify columns
    if mode == '3axis': cols = ['acc_x', 'acc_y', 'acc_z']
    elif mode == '6axis': cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    elif mode == '9axis': cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z']

    # Gather all files
    categories = {'adl': 0, 'fall': 1}
    all_tasks = []
    for cat, label in categories.items():
        files = glob.glob(os.path.join(DATA_PATH, cat, "**", "*.csv"), recursive=True)
        for f in files:
            all_tasks.append((f, label))

    print(f"Total files: {len(all_tasks)}")
    
    # Process in Batches
    X_imu_buffer, X_hr_buffer, y_buffer = [], [], []
    part_idx = 0

    for i, (f_path, label) in enumerate(all_tasks):
        fused_df, hrt_data = parse_file(f_path, mode)
        
        if fused_df is not None and len(fused_df) >= WINDOW_SIZE:
            imu_vals = fused_df[['t'] + cols].values
            
            # Sliding Window
            for j in range(0, len(imu_vals) - WINDOW_SIZE, STEP_SIZE):
                window = imu_vals[j : j + WINDOW_SIZE]
                
                # HR Sync
                start_t, end_t = window[0, 0], window[-1, 0]
                avg_hr = 75.0
                if not hrt_data.empty:
                    rel_hr = hrt_data[(hrt_data['t'] >= start_t) & (hrt_data['t'] <= end_t)]
                    if not rel_hr.empty:
                        avg_hr = rel_hr['bpm'].mean()
                    else:
                        avg_hr = hrt_data['bpm'].mean()
                
                if np.isnan(avg_hr): avg_hr = 75.0

                X_imu_buffer.append(window[:, 1:]) # Drop time col
                X_hr_buffer.append(avg_hr)
                y_buffer.append(label)

        # Print progress
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(all_tasks)} files...")

        # Save Batch if buffer is full or last file
        if len(X_imu_buffer) >= 2000 or (i == len(all_tasks) - 1 and len(X_imu_buffer) > 0):
            save_path = os.path.join(OUTPUT_DIR, f"{mode}_part_{part_idx}.npz")
            
            np.savez_compressed(
                save_path,
                X_imu=np.array(X_imu_buffer, dtype=np.float32),
                X_hr=np.array(X_hr_buffer, dtype=np.float32),
                y=np.array(y_buffer, dtype=np.int32)
            )
            print(f"Saved {save_path} (Windows: {len(X_imu_buffer)})")
            
            # Clear RAM
            X_imu_buffer, X_hr_buffer, y_buffer = [], [], []
            part_idx += 1
            gc.collect()

# Run
process_and_save('3axis')
process_and_save('6axis')
process_and_save('9axis')