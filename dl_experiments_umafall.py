import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import argparse
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

print(f"Python Version: {sys.version}", flush=True)
print("Starting dl_experiments_umafall.py script...", flush=True)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None, help='Specific model to run (e.g., GRU_Only, BiGRU_Only)')
parser.add_argument('--data_dir', type=str, default='./processed_tensors_umafall_no_hr')
parser.add_argument('--output_dir', type=str, default='./results_dl_umafall')
args_cmd = parser.parse_args()

# ==========================================
# 1. Configuration
# ==========================================
DATA_DIR = './processed_tensors_umafall_no_hr'
OUTPUT_DIR = './results_dl_umafall'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODES = ['3axis', '6axis', '9axis']
EPOCHS = 100
BATCH_SIZE = 64

# ==========================================
# 2. Model Builders (IMU ONLY)
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

# ==========================================
# 3. Performance Profiling
# ==========================================

def get_flops(model):
    concrete_func = tf.function(lambda x: model(x))
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

def measure_latency(model, window_size, n_channels, n_iterations=100):
    imu_dummy = np.random.randn(1, window_size, n_channels).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        _ = model.predict(imu_dummy, verbose=0)
    
    start_time = time.time()
    for _ in range(n_iterations):
        _ = model.predict(imu_dummy, verbose=0)
    end_time = time.time()
    
    avg_latency_ms = ((end_time - start_time) / n_iterations) * 1000
    return avg_latency_ms

# ==========================================
# 3. Experiment Loop
# ==========================================

MODEL_BUILDERS = {
    'CNN_Only': build_cnn_only_model,
    'LSTM_Only': build_lstm_only_model,
    'CNN_LSTM_Unidirectional': build_unidirectional_cnn_lstm_model,
    'Transformer': build_transformer_model,
    'CNN_BiLSTM': build_cnn_bilstm_model,
    'CNN_GRU': build_cnn_gru_model,
    'CNN_BiGRU': build_cnn_bigru_model,
    'GRU_Only': build_gru_only_model,
    'BiGRU_Only': build_bigru_only_model,
    'TCN': build_tcn_model,
    'MultiScale_SE_BiLSTM': build_multiscale_se_bilstm_model,
}

if args_cmd.model:
    if args_cmd.model in MODEL_BUILDERS:
        MODEL_BUILDERS = {args_cmd.model: MODEL_BUILDERS[args_cmd.model]}
    else:
        print(f"Model {args_cmd.model} not found in MODEL_BUILDERS. Available models: {list(MODEL_BUILDERS.keys())}")
        sys.exit(1)

def run_experiments():
    results_summary = []

    for mode in MODES:
        data_path = os.path.join(DATA_DIR, f"data_{mode}_no_hr.npz")
        if not os.path.exists(data_path):
            print(f"Data file {data_path} not found! Skipping...", flush=True)
            continue
        
        print(f"\n{'='*40}", flush=True)
        print(f"Processing Mode: {mode}", flush=True)
        print(f"{'='*40}", flush=True)
        
        data = np.load(data_path)
        X_imu = data['X_imu']
        y = data['y']
        
        # Split
        X_imu_train, X_imu_test, y_train, y_test = train_test_split(
            X_imu, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scaling IMU
        scaler_imu = StandardScaler()
        N_train, T, F = X_imu_train.shape
        X_imu_train = scaler_imu.fit_transform(X_imu_train.reshape(-1, F)).reshape(N_train, T, F)
        N_test = X_imu_test.shape[0]
        X_imu_test = scaler_imu.transform(X_imu_test.reshape(-1, F)).reshape(N_test, T, F)

        case_label = f"{mode}_no_HR"
        print(f"\n--- Case: {case_label} ---", flush=True)

        for model_name, builder_fn in MODEL_BUILDERS.items():
            print(f"Building and Training: {model_name}...", flush=True)
            model = builder_fn(window_size=T, n_channels=F)
            
            model.compile(
                optimizer=optimizers.Adam(learning_rate=1e-4),
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )

            # Train
            early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
            
            history = model.fit(
                x=X_imu_train,
                y=y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(X_imu_test, y_test),
                callbacks=[early_stop],
                verbose=0 
            )

            # Final Evaluation
            y_pred_prob = model.predict(X_imu_test, verbose=0)
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                tn = fp = fn = tp = 0
                sens = spec = 0

            print(f"  Result: {model_name} | Acc: {acc:.4f} | Sens: {sens:.4f} | Spec: {spec:.4f}", flush=True)
            
            results_summary.append({
                'Case': case_label,
                'Model': model_name,
                'Accuracy': acc,
                'Sensitivity': sens,
                'Specificity': spec,
                'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
                'Params': model.count_params(),
                'Size_KB': (model.count_params() * 4) / 1024,
                'FLOPs': get_flops(model),
                'Latency_ms': measure_latency(model, T, F)
            })

    # Save final results
    import pandas as pd
    df = pd.DataFrame(results_summary)
    df.to_csv(f"{OUTPUT_DIR}/dl_comparison.csv", index=False)
    with open(f"{OUTPUT_DIR}/dl_comparison.txt", "w") as f:
        f.write("UMAFall DL Experiment Results Summary\n")
        f.write("="*60 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n" + "="*60 + "\n")
    
    print(f"\nExperiments completed. Results saved in {OUTPUT_DIR}", flush=True)

if __name__ == "__main__":
    run_experiments()
