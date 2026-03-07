import os
import glob
import numpy as np
import gc

DATA_PATH = "UMAFall_Dataset/UMAFall_Dataset"
OUTPUT_DIR = "processed_tensors_umafall_no_hr"
WINDOW_SIZE = 128
STEP_SIZE = 64

os.makedirs(OUTPUT_DIR, exist_ok=True)

class LoadUMAFallDatasetPure:
    def read_umafall_csv(self, path):
        data = []
        with open(path, "r", errors="ignore") as f:
            for line in f:
                if line.startswith("%") or not line.strip():
                    continue
                parts = line.strip().replace(",", ".").split(";")
                if len(parts) >= 7:
                    try:
                        x = float(parts[2])
                        y = float(parts[3])
                        z = float(parts[4])
                        sensor_id = int(parts[5])
                        loc_id = int(parts[6])
                        data.append([x, y, z, sensor_id, loc_id])
                    except ValueError:
                        continue
        return np.array(data, dtype=np.float32)

    def extract_imu(self, df_array, mode, location_id=None,
                        acc_id=0, gyro_id=1, mag_id=2):
        if df_array.shape[0] == 0:
            raise ValueError("Empty data array.")
            
        sensors_needed = [acc_id]
        if mode in ['6axis', '9axis']:
            sensors_needed.append(gyro_id)
        if mode == '9axis':
            sensors_needed.append(mag_id)
            
        unique_locs = np.unique(df_array[:, 4]).astype(int)

        if location_id is None:
            valid_locs = []
            for loc in unique_locs:
                ok = True
                for sid in sensors_needed:
                    mask = (df_array[:, 3] == sid) & (df_array[:, 4] == loc)
                    if not np.any(mask):
                        ok = False
                        break
                if ok:
                    valid_locs.append(loc)
            if not valid_locs:
                raise ValueError(f"No location has required sensors for {mode}.")
            location_id = valid_locs[0]

        extracted = []
        for sid in sensors_needed:
            mask = (df_array[:, 3] == sid) & (df_array[:, 4] == location_id)
            sensor_data = df_array[mask][:, 0:3] 
            extracted.append(sensor_data)

        n_min = min(len(s) for s in extracted)
        if n_min == 0:
            raise ValueError("No valid samples found after masking.")
            
        truncated = [s[:n_min] for s in extracted]
        return np.concatenate(truncated, axis=1)

    def parse_umafall_filename(self, path):
        base = os.path.basename(path)
        name, _ = os.path.splitext(base)
        parts = name.split("_")
        if len(parts) < 6:
            raise ValueError(f"Unexpected UMAFall filename format: {base}")
            
        movement_type = parts[3]
        fall_label = 1 if movement_type.upper() == "FALL" else 0
        return {"fall_label": fall_label}

def create_dataset(mode):
    print(f"--- Generating UMAFall Dataset for {mode} ---")
    X_imu_list = []
    y_list = []
    loader = LoadUMAFallDatasetPure()
    
    files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
    if not files:
        print(f"ERROR: No files found in {DATA_PATH}.")
        return

    processed_count = 0
    failed_count = 0
    
    # Process just the first 10 files for test validation mode
    for i, f in enumerate(files):
        if i > 0 and i % 50 == 0:
            print(f"  Processed {i}/{len(files)} files...")
            gc.collect()
            
        try:
            info = loader.parse_umafall_filename(f)
            df_array = loader.read_umafall_csv(f)
            
            loc_id = 1
            try:
                imu_vals = loader.extract_imu(df_array, mode=mode, location_id=loc_id)
            except ValueError:
                try:
                    imu_vals = loader.extract_imu(df_array, mode=mode, location_id=None)
                except ValueError:
                    failed_count += 1
                    continue
            
            if len(imu_vals) < WINDOW_SIZE:
                continue

            for j in range(0, len(imu_vals) - WINDOW_SIZE + 1, STEP_SIZE):
                window_imu = imu_vals[j : j + WINDOW_SIZE]
                X_imu_list.append(window_imu)
                y_list.append(info['fall_label'])
                
            processed_count += 1

        except Exception as e:
            failed_count += 1
            print(f'Error on {f}: {e}')
            continue

    print(f"Successfully processed {processed_count} files, failed on {failed_count} files.")
    if len(X_imu_list) == 0:
        print(f"ERROR: No valid data generated for {mode}.")
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
