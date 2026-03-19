import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras_tuner as kt

import random
def set_random_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
set_random_seed()

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='9axis', help='Which dataset to tune on')
parser.add_argument('--data_dir', type=str, default='./processed_tensors')
parser.add_argument('--tuner_dir', type=str, default='./hpo_logs')
args = parser.parse_args()

os.makedirs(args.tuner_dir, exist_ok=True)
print(f"--- Starting Hyperparameter Tuning for {args.mode} ---")

# 1. Load Data (Using your Fused Model Data)
data_path = os.path.join(args.data_dir, f"data_{args.mode}.npz")
data = np.load(data_path)
X_imu = data['X_imu']
X_hr = data['X_hr'].reshape(-1, 1)
y = data['y']

indices = np.arange(len(y))
X_imu_train, X_imu_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_imu, y, indices, test_size=0.2, random_state=42, stratify=y
)
X_hr_train, X_hr_test = X_hr[idx_train], X_hr[idx_test]

# 2. Scale Data
scaler_imu = StandardScaler()
N_train, T, F = X_imu_train.shape
X_imu_train = scaler_imu.fit_transform(X_imu_train.reshape(-1, F)).reshape(N_train, T, F)
X_imu_test = scaler_imu.transform(X_imu_test.reshape(-1, F)).reshape(X_imu_test.shape[0], T, F)

scaler_hr = StandardScaler()
X_hr_train = scaler_hr.fit_transform(X_hr_train)
X_hr_test = scaler_hr.transform(X_hr_test)

# 3. Define the Hypermodel
def build_model(hp):
    imu_input = layers.Input(shape=(T, F), name="imu_input")

    # Tune CNN Filters
    filters_1 = hp.Int('conv1_filters', min_value=64, max_value=256, step=64)
    x = layers.Conv1D(filters_1, kernel_size=3, padding="same", activation="relu")(imu_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    filters_2 = hp.Int('conv2_filters', min_value=64, max_value=256, step=64)
    x = layers.Conv1D(filters_2, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # Tune LSTM Units
    lstm_units = hp.Int('lstm_units', min_value=64, max_value=256, step=64)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False))(x)

    # HR Branch
    hr_input = layers.Input(shape=(1,), name="hr_input")
    y_branch = layers.Dense(16, activation="relu")(hr_input)

    # Fusion
    combined = layers.Concatenate()([x, y_branch])

    # Tune Dense Layer and Dropout
    dense_units = hp.Int('dense_units', min_value=32, max_value=128, step=32)
    z = layers.Dense(dense_units, activation="relu")(combined)
    
    dropout_rate = hp.Float('dropout', min_value=0.2, max_value=0.6, step=0.1)
    z = layers.Dropout(dropout_rate)(z)
    outputs = layers.Dense(1, activation="sigmoid")(z)

    model = models.Model(inputs=[imu_input, hr_input], outputs=outputs)
    
    # Tune Learning Rate
    lr = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# 4. Initialize Hyperband Tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=50,          # Max epochs per model
    factor=3,               # Reduction factor for Hyperband
    directory=args.tuner_dir,
    project_name=f"fall_detect_{args.mode}_tuning"
)

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 5. Start Search
print("Starting search...")
tuner.search(
    x=[X_imu_train, X_hr_train],
    y=y_train,
    epochs=50,
    batch_size=64,
    validation_data=([X_imu_test, X_hr_test], y_test),
    callbacks=[early_stop],
    verbose=2
)

# 6. Extract Best Results
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\n=========================================")
print("OPTIMAL HYPERPARAMETERS FOUND:")
print(f"Conv1 Filters:  {best_hps.get('conv1_filters')}")
print(f"Conv2 Filters:  {best_hps.get('conv2_filters')}")
print(f"LSTM Units:     {best_hps.get('lstm_units')}")
print(f"Dense Units:    {best_hps.get('dense_units')}")
print(f"Dropout Rate:   {best_hps.get('dropout')}")
print(f"Learning Rate:  {best_hps.get('learning_rate')}")
print("=========================================\n")

# Save to text file
with open(f"{args.tuner_dir}/best_hps_{args.mode}.txt", "w") as f:
    f.write(f"Conv1: {best_hps.get('conv1_filters')}\n")
    f.write(f"Conv2: {best_hps.get('conv2_filters')}\n")
    f.write(f"LSTM: {best_hps.get('lstm_units')}\n")
    f.write(f"Dense: {best_hps.get('dense_units')}\n")
    f.write(f"Dropout: {best_hps.get('dropout')}\n")
    f.write(f"LR: {best_hps.get('learning_rate')}\n")