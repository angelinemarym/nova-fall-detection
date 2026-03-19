import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import argparse
import time
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

# ==========================================
# 1. Configuration
# ==========================================
DATA_DIR = './processed_tensors_hifd'
OUTPUT_DIR = './results_dl_hifd'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODES = ['3axis', '6axis']
USE_HR_OPTIONS = [False, True]
EPOCHS = 50 
BATCH_SIZE = 64

# ==========================================
# 2. Model Builders
# ==========================================

from dl_models import (
    build_cnn_only_model,
    build_lstm_only_model,
    build_unidirectional_cnn_lstm_model,
    build_transformer_model,
    build_cnn_bilstm_model,
    build_cnn_gru_model,
    build_cnn_bigru_model,
    build_gru_only_model,
    build_bigru_only_model,
    build_tcn_model,
    build_multiscale_se_bilstm_model
)

MODEL_BUILDERS = {
    'CNN_Only': build_cnn_only_model,
    'LSTM_Only': build_lstm_only_model,
    'CNN_LSTM': build_unidirectional_cnn_lstm_model,
    'Transformer': build_transformer_model,
    'CNN_BiLSTM': build_cnn_bilstm_model,
    'CNN_GRU': build_cnn_gru_model,
    'CNN_BiGRU': build_cnn_bigru_model,
    'GRU_Only': build_gru_only_model,
    'BiGRU_Only': build_bigru_only_model,
    'TCN': build_tcn_model,
    'MultiScale_SE_BiLSTM': build_multiscale_se_bilstm_model,
}

# ==========================================
# 3. Performance Profiling
# ==========================================

def get_flops(model):
    try:
        concrete_func = tf.function(lambda x: model(x))
        if isinstance(model.input_shape, list):
            input_shapes = [tf.TensorSpec([1] + list(s[1:]), tf.float32) for s in model.input_shape]
            concrete_func = concrete_func.get_concrete_function(input_shapes)
        else:
            concrete_func = concrete_func.get_concrete_function(
                tf.TensorSpec([1] + list(model.input_shape[1:]), tf.float32)
            )
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            return flops.total_float_ops if flops is not None else 0
    except:
        return 0

def measure_latency(model, window_size, n_channels, use_hr, n_iterations=50):
    imu_dummy = np.random.randn(1, window_size, n_channels).astype(np.float32)
    hr_dummy = np.random.randn(1, 1).astype(np.float32)
    inputs = [imu_dummy, hr_dummy] if use_hr else [imu_dummy]
    for _ in range(5):
        _ = model.predict(inputs, verbose=0)
    start_time = time.time()
    for _ in range(n_iterations):
        _ = model.predict(inputs, verbose=0)
    end_time = time.time()
    return ((end_time - start_time) / n_iterations) * 1000

# ==========================================
# 4. Main Experiment Loop
# ==========================================

def run_experiments():
    results_summary = []

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
            
            X_imu = data['X_imu']
            X_hr = data['X_hr']
            y = data['y']
            
            if len(y) == 0:
                print(f"ERROR: {data_path} contains zero samples. Skipping...")
                sys.exit(1)
                
            X_hr = X_hr.reshape(-1, 1)
        except Exception as e:
            print(f"ERROR loading {data_path}: {e}")
            sys.exit(1)


        
        indices = np.arange(len(y))
        X_imu_train, X_imu_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X_imu, y, indices, test_size=0.2, random_state=42, stratify=y
        )
        X_hr_train = X_hr[idx_train]
        X_hr_test = X_hr[idx_test]

        scaler_imu = StandardScaler()
        N_train, T, F = X_imu_train.shape
        X_imu_train = scaler_imu.fit_transform(X_imu_train.reshape(-1, F)).reshape(N_train, T, F)
        N_test = X_imu_test.shape[0]
        X_imu_test = scaler_imu.transform(X_imu_test.reshape(-1, F)).reshape(N_test, T, F)

        scaler_hr = StandardScaler()
        X_hr_train = scaler_hr.fit_transform(X_hr_train)
        X_hr_test = scaler_hr.transform(X_hr_test)

        for use_hr in USE_HR_OPTIONS:
            case_label = f"{mode}_with_HR" if use_hr else f"{mode}_no_HR"
            print(f"--- Case: {case_label} ---")

            for model_name, builder_fn in MODEL_BUILDERS.items():
                print(f"Training {model_name}...")
                model = builder_fn(T, F, use_hr)
                model.compile(optimizer=optimizers.Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])

                early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
                train_x = [X_imu_train, X_hr_train] if use_hr else [X_imu_train]
                val_x = [X_imu_test, X_hr_test] if use_hr else [X_imu_test]
                
                class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
                class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

                model.fit(train_x, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(val_x, y_test), callbacks=[early_stop], class_weight=class_weight_dict, verbose=0)

                y_pred_prob = model.predict(val_x, verbose=0)
                y_pred = (y_pred_prob > 0.5).astype(int).flatten()
                acc = accuracy_score(y_test, y_pred)
                
                # Robust Metric Calculation
                tp = np.sum((y_test == 1) & (y_pred == 1))
                tn = np.sum((y_test == 0) & (y_pred == 0))
                fp = np.sum((y_test == 0) & (y_pred == 1))
                fn = np.sum((y_test == 1) & (y_pred == 0))
                
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0


                results_summary.append({
                    'Dataset': 'HIFD',
                    'Case': case_label,
                    'Model': model_name,
                    'Accuracy': acc,
                    'Sensitivity': sens,
                    'Specificity': spec,
                    'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
                    'Params': model.count_params(),
                    'Size_KB': (model.count_params() * 4) / 1024,
                    'FLOPs': get_flops(model),
                    'Latency_ms': measure_latency(model, T, F, use_hr)
                })

    import pandas as pd
    df = pd.DataFrame(results_summary)
    df.to_csv(f"{OUTPUT_DIR}/dl_comparison_hifd.csv", index=False)
    print(f"\nExperiments completed. Results saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    run_experiments()
