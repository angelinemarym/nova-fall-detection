import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import joblib
import sys
import time

# ==========================================
# 1. Configuration
# ==========================================
DATA_DIR = './processed_tensors_hifd'
OUTPUT_DIR = './results_ml_hifd'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODES = ['3axis', '6axis']
USE_HR_OPTIONS = [False, True]

MODELS = {
    'kNN': KNeighborsClassifier(n_neighbors=5),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'NaiveBayes': GaussianNB(),
    'ANN': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# ==========================================
# 2. Feature Extraction
# ==========================================
def extract_features(X_imu):
    N, T, F = X_imu.shape
    f_mean = np.mean(X_imu, axis=1)
    f_std = np.std(X_imu, axis=1)
    f_min = np.min(X_imu, axis=1)
    f_max = np.max(X_imu, axis=1)
    f_median = np.median(X_imu, axis=1)
    return np.hstack([f_mean, f_std, f_min, f_max, f_median])

def get_model_size_kb(model_name, model, X_train, y_train):
    if model_name == 'kNN':
        size_bytes = X_train.nbytes + y_train.nbytes
    else:
        temp_path = 'temp_model_hifd.pkl'
        joblib.dump(model, temp_path)
        size_bytes = os.path.getsize(temp_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return size_bytes / 1024

def measure_ml_latency(model, X_sample, n_iterations=100):
    for _ in range(10):
        _ = model.predict(X_sample)
    start_time = time.time()
    for _ in range(n_iterations):
        _ = model.predict(X_sample)
    end_time = time.time()
    return ((end_time - start_time) / n_iterations) * 1000

# ==========================================
# 3. Main Experiment Loop
# ==========================================
def run_experiments():
    summary_results = []

    for mode in MODES:
        data_path = os.path.join(DATA_DIR, f"data_hifd_{mode}.npz")
        if not os.path.exists(data_path):
            print(f"ERROR: Data file {data_path} not found! Run hifd_preprocess.py first.")
            sys.exit(1)
        
        print(f"\nProcessing HIFD Mode: {mode}")
        try:
            data = np.load(data_path)
            if 'X_imu' not in data or 'y' not in data:
                print(f"ERROR: {data_path} is missing required keys. Found: {list(data.keys())}")
                sys.exit(1)
            
            X_imu_raw = data['X_imu']
            X_hr_raw = data['X_hr']
            y = data['y']
            
            if len(y) == 0:
                print(f"ERROR: {data_path} contains zero samples. Skipping...")
                sys.exit(1)
                
            X_imu_feats = extract_features(X_imu_raw)
        except Exception as e:
            print(f"ERROR loading {data_path}: {e}")
            sys.exit(1)


        
        for use_hr in USE_HR_OPTIONS:
            case_name = f"{mode}_with_HR" if use_hr else f"{mode}_no_HR"
            print(f"--- Case: {case_name} ---")
            
            if use_hr:
                X = np.hstack([X_imu_feats, X_hr_raw.reshape(-1, 1)])
            else:
                X = X_imu_feats
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # --- Handle Class Imbalance via Random Undersampling ---
            fall_indices = np.where(y_train == 1)[0]
            non_fall_indices = np.where(y_train == 0)[0]
            
            if len(fall_indices) > 0 and len(non_fall_indices) > len(fall_indices):
                np.random.seed(42)
                undersampled_non_fall_indices = np.random.choice(non_fall_indices, size=len(fall_indices), replace=False)
                balanced_indices = np.concatenate([fall_indices, undersampled_non_fall_indices])
                np.random.shuffle(balanced_indices)
                
                X_train = X_train[balanced_indices]
                y_train = y_train[balanced_indices]
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            for model_name, model in MODELS.items():
                print(f"Training {model_name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                
                # Robust Metric Calculation
                tp = np.sum((y_test == 1) & (y_pred == 1))
                tn = np.sum((y_test == 0) & (y_pred == 0))
                fp = np.sum((y_test == 0) & (y_pred == 1))
                fn = np.sum((y_test == 1) & (y_pred == 0))
                
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0

                
                summary_results.append({
                    'Dataset': 'HIFD',
                    'Case': case_name,
                    'Model': model_name,
                    'Accuracy': acc,
                    'Sensitivity': sens,
                    'Specificity': spec,
                    'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
                    'Size_KB': get_model_size_kb(model_name, model, X_train, y_train),
                    'Latency_ms': measure_ml_latency(model, X_test[0:1])
                })

    df = pd.DataFrame(summary_results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'summary_results_hifd.csv'), index=False)
    print(f"\nExperiments completed. Results saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    run_experiments()
