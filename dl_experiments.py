import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

print(f"Python Version: {sys.version}", flush=True)
print("Starting dl_experiments.py script...", flush=True)

# ==========================================
# 1. Configuration
# ==========================================
DATA_DIR = './processed_tensors'
OUTPUT_DIR = './results_dl'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODES = ['3axis', '6axis', '9axis']
USE_HR_OPTIONS = [False, True]
EPOCHS = 100
BATCH_SIZE = 64

# ==========================================
# 2. Model Builders
# ==========================================

def build_cnn_only_model(window_size, n_channels, use_hr):
    imu_input = layers.Input(shape=(window_size, n_channels), name="imu_input")
    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(imu_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x) # Replace LSTM with Global Pool

    if use_hr:
        hr_input = layers.Input(shape=(1,), name="hr_input")
        hr_branch = layers.Dense(16, activation="relu")(hr_input)
        combined = layers.Concatenate()([x, hr_branch])
        inputs = [imu_input, hr_input]
    else:
        combined = x
        inputs = [imu_input]

    z = layers.Dense(96, activation="relu")(combined)
    z = layers.Dropout(0.4)(z)
    outputs = layers.Dense(1, activation="sigmoid")(z)
    return models.Model(inputs=inputs, outputs=outputs, name="CNN_Only")

def build_lstm_only_model(window_size, n_channels, use_hr):
    imu_input = layers.Input(shape=(window_size, n_channels), name="imu_input")
    # No CNN layers
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(imu_input)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)

    if use_hr:
        hr_input = layers.Input(shape=(1,), name="hr_input")
        hr_branch = layers.Dense(16, activation="relu")(hr_input)
        combined = layers.Concatenate()([x, hr_branch])
        inputs = [imu_input, hr_input]
    else:
        combined = x
        inputs = [imu_input]

    z = layers.Dense(96, activation="relu")(combined)
    z = layers.Dropout(0.4)(z)
    outputs = layers.Dense(1, activation="sigmoid")(z)
    return models.Model(inputs=inputs, outputs=outputs, name="LSTM_Only")

def build_unidirectional_cnn_lstm_model(window_size, n_channels, use_hr):
    imu_input = layers.Input(shape=(window_size, n_channels), name="imu_input")
    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(imu_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    # Unidirectional LSTM instead of BiLSTM
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=False)(x)

    if use_hr:
        hr_input = layers.Input(shape=(1,), name="hr_input")
        hr_branch = layers.Dense(16, activation="relu")(hr_input)
        combined = layers.Concatenate()([x, hr_branch])
        inputs = [imu_input, hr_input]
    else:
        combined = x
        inputs = [imu_input]

    z = layers.Dense(96, activation="relu")(combined)
    z = layers.Dropout(0.4)(z)
    outputs = layers.Dense(1, activation="sigmoid")(z)
    return models.Model(inputs=inputs, outputs=outputs, name="CNN_LSTM_Unidirectional")

def build_transformer_model(window_size, n_channels, use_hr):
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

    imu_input = layers.Input(shape=(window_size, n_channels), name="imu_input")
    # Lightweight Transformer block
    x = transformer_encoder(imu_input, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    x = layers.GlobalAveragePooling1D()(x)

    if use_hr:
        hr_input = layers.Input(shape=(1,), name="hr_input")
        hr_branch = layers.Dense(16, activation="relu")(hr_input)
        combined = layers.Concatenate()([x, hr_branch])
        inputs = [imu_input, hr_input]
    else:
        combined = x
        inputs = [imu_input]

    z = layers.Dense(96, activation="relu")(combined)
    z = layers.Dropout(0.4)(z)
    outputs = layers.Dense(1, activation="sigmoid")(z)
    return models.Model(inputs=inputs, outputs=outputs, name="Transformer")

# ==========================================
# 3. Experiment Loop
# ==========================================

MODEL_BUILDERS = {
    'CNN_Only': build_cnn_only_model,
    'LSTM_Only': build_lstm_only_model,
    'CNN_LSTM_Unidirectional': build_unidirectional_cnn_lstm_model,
    'Transformer': build_transformer_model
}

def run_experiments():
    results_summary = []

    for mode in MODES:
        data_path = os.path.join(DATA_DIR, f"data_{mode}.npz")
        if not os.path.exists(data_path):
            print(f"Data file {data_path} not found! Skipping...", flush=True)
            continue
        
        print(f"\n{'='*40}", flush=True)
        print(f"Processing Mode: {mode}", flush=True)
        print(f"{'='*40}", flush=True)
        
        data = np.load(data_path)
        X_imu = data['X_imu']
        X_hr = data['X_hr'].reshape(-1, 1)
        y = data['y']
        
        # Split
        indices = np.arange(len(y))
        X_imu_train, X_imu_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X_imu, y, indices, test_size=0.2, random_state=42, stratify=y
        )
        X_hr_train = X_hr[idx_train]
        X_hr_test = X_hr[idx_test]

        # Scaling IMU
        scaler_imu = StandardScaler()
        N_train, T, F = X_imu_train.shape
        X_imu_train = scaler_imu.fit_transform(X_imu_train.reshape(-1, F)).reshape(N_train, T, F)
        N_test = X_imu_test.shape[0]
        X_imu_test = scaler_imu.transform(X_imu_test.reshape(-1, F)).reshape(N_test, T, F)

        # Scaling HR
        scaler_hr = StandardScaler()
        X_hr_train = scaler_hr.fit_transform(X_hr_train)
        X_hr_test = scaler_hr.transform(X_hr_test)

        for use_hr in USE_HR_OPTIONS:
            case_label = f"{mode}_with_HR" if use_hr else f"{mode}_no_HR"
            print(f"\n--- Case: {case_label} ---", flush=True)

            for model_name, builder_fn in MODEL_BUILDERS.items():
                print(f"Building and Training: {model_name}...", flush=True)
                model = builder_fn(window_size=T, n_channels=F, use_hr=use_hr)
                
                model.compile(
                    optimizer=optimizers.Adam(learning_rate=1e-4),
                    loss="binary_crossentropy",
                    metrics=["accuracy"]
                )

                # Train
                early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
                
                train_x = [X_imu_train, X_hr_train] if use_hr else [X_imu_train]
                val_x = [X_imu_test, X_hr_test] if use_hr else [X_imu_test]
                
                history = model.fit(
                    x=train_x,
                    y=y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(val_x, y_test),
                    callbacks=[early_stop],
                    verbose=0 # Quiet training for summary
                )

                # Final Evaluation
                y_pred_prob = model.predict(val_x, verbose=0)
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
                    'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
                })
                
                # Save small model if needed (skipping for now to save HPC space)
                # model.save(f"{OUTPUT_DIR}/model_{case_label}_{model_name}.h5")

    # Save final results
    import pandas as pd
    df = pd.DataFrame(results_summary)
    df.to_csv(f"{OUTPUT_DIR}/dl_comparison.csv", index=False)
    with open(f"{OUTPUT_DIR}/dl_comparison.txt", "w") as f:
        f.write("Advanced DL Experiment Results\n")
        f.write("="*60 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n" + "="*60 + "\n")
    
    print(f"\nExperiments completed. Results saved in {OUTPUT_DIR}", flush=True)

if __name__ == "__main__":
    run_experiments()
