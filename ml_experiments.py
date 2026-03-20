import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import joblib
import sys
import time

import random
def set_random_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
set_random_seed()

print(f"Python Version: {sys.version}", flush=True)
print("Starting ml_experiments.py script...", flush=True)

# ==========================================
# 1. Configuration
# ==========================================
DATA_DIR = './processed_tensors'
OUTPUT_DIR = './results_ml'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODES = ['3axis', '6axis', '9axis']
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
    """
    Extracts features from raw IMU windows.
    X_imu shape: (N, window_size, n_channels)
    Returns: (N, n_channels * 5)
    """
    N, T, F = X_imu.shape
    features = []
    
    # Statistical features per channel
    f_mean = np.mean(X_imu, axis=1)    # (N, F)
    f_std = np.std(X_imu, axis=1)      # (N, F)
    f_min = np.min(X_imu, axis=1)      # (N, F)
    f_max = np.max(X_imu, axis=1)      # (N, F)
    f_median = np.median(X_imu, axis=1) # (N, F)
    
    # Concatenate all features
    X_feats = np.hstack([f_mean, f_std, f_min, f_max, f_median])
    return X_feats

# ==========================================
# 3. Performance Profiling
# ==========================================

def get_model_size_kb(model_name, model, X_train, y_train):
    """
    Estimates model size in KB.
    """
    if model_name == 'kNN':
        # kNN stores the entire training set
        size_bytes = X_train.nbytes + y_train.nbytes
    else:
        # Save to a temporary file and check size
        temp_path = 'temp_model.pkl'
        joblib.dump(model, temp_path)
        size_bytes = os.path.getsize(temp_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return size_bytes / 1024

def measure_ml_latency(model, X_sample, n_iterations=1000):
    """
    Measures average inference latency in ms.
    """
    # X_sample should be a single row (1, n_features)
    # Warmup
    for _ in range(10):
        _ = model.predict(X_sample)
        
    start_time = time.time()
    for _ in range(n_iterations):
        _ = model.predict(X_sample)
    end_time = time.time()
    
    avg_latency_ms = ((end_time - start_time) / n_iterations) * 1000
    return avg_latency_ms

# ==========================================
# 3. Main Experiment Loop
# ==========================================
def run_experiments():
    summary_results = []

    for mode in MODES:
        data_path = os.path.join(DATA_DIR, f"data_{mode}.npz")
        if not os.path.exists(data_path):
            print(f"Data file {data_path} not found! Skipping...", flush=True)
            continue
        
        print(f"\n{'='*40}", flush=True)
        print(f"Processing Mode: {mode}", flush=True)
        print(f"{'='*40}", flush=True)
        
        data = np.load(data_path)
        X_imu_raw = data['X_imu']
        X_hr_raw = data['X_hr']
        y = data['y']
        
        # Feature Extraction
        print("Extracting features...", flush=True)
        X_imu_feats = extract_features(X_imu_raw)
        
        for use_hr in USE_HR_OPTIONS:
            case_name = f"{mode}_with_HR" if use_hr else f"{mode}_no_HR"
            print(f"\n--- Case: {case_name} ---", flush=True)
            
            # Combine IMU features and HR
            if use_hr:
                X = np.hstack([X_imu_feats, X_hr_raw.reshape(-1, 1)])
            else:
                X = X_imu_feats
            
            # Split Data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scaling
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Train and Evaluate Models
            for model_name, model in MODELS.items():
                print(f"Training {model_name}...", flush=True)
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Metrics
                acc = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                
                # Avoid errors if cm is not 2x2
                if cm.size == 4:
                    tn, fp, fn, tp = cm.ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                else:
                    tn = fp = fn = tp = 0
                    sensitivity = specificity = 0
                
                print(f"  {model_name} Accuracy: {acc:.4f}", flush=True)
                
                # Log result
                res = {
                    'Case': case_name,
                    'Model': model_name,
                    'Accuracy': acc,
                    'Sensitivity': sensitivity,
                    'Specificity': specificity,
                    'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
                    'Size_KB': get_model_size_kb(model_name, model, X_train, y_train),
                    'Latency_ms': measure_ml_latency(model, X_test[0:1])
                }
                summary_results.append(res)
                
                # Save model
                model_save_path = os.path.join(OUTPUT_DIR, f"model_{case_name}_{model_name}.pkl")
                joblib.dump(model, model_save_path)

    # Save summary to CSV and TXT
    df_results = pd.DataFrame(summary_results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'summary_results.csv'), index=False)
    
    with open(os.path.join(OUTPUT_DIR, 'summary_results.txt'), 'w') as f:
        f.write("Traditional ML Experiment Results Summary\n")
        f.write("="*50 + "\n")
        f.write(df_results.to_string(index=False))
        f.write("\n" + "="*50 + "\n")

    print(f"\nExperiments completed. Results saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    run_experiments()
