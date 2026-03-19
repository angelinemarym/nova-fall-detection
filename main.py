import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import random
def set_random_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
set_random_seed()

# ==========================================
# 1. Setup
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='6axis', choices=['3axis', '6axis', '9axis'])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--data_dir', type=str, default='./processed_tensors')
parser.add_argument('--output_dir', type=str, default='./results')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
print(f"Running Experiment: {args.mode}")

# ==========================================
# 2. Load Data
# ==========================================
data_path = os.path.join(args.data_dir, f"data_{args.mode}.npz")
if not os.path.exists(data_path):
    print(f"Data file {data_path} not found! Run preprocess.py first.")
    exit(1)

data = np.load(data_path)
X_imu = data['X_imu']
X_hr = data['X_hr']
y = data['y']

# Handle HR Shape: Needs to be (N, 1) for StandardScaler and Model Input
X_hr = X_hr.reshape(-1, 1)

print(f"Data Loaded - IMU: {X_imu.shape}, HR: {X_hr.shape}, Labels: {y.shape}")

# Split
# Note: We split indices first to ensure IMU and HR stay aligned
indices = np.arange(len(y))
X_imu_train, X_imu_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_imu, y, indices, test_size=0.2, random_state=42, stratify=y
)
X_hr_train = X_hr[idx_train]
X_hr_test = X_hr[idx_test]

# ==========================================
# 3. Scaling (As per your code)
# ==========================================
# Scale IMU: Reshape -> Fit -> Reshape
scaler_imu = StandardScaler()
N_train, T, F = X_imu_train.shape
X_imu_train = scaler_imu.fit_transform(X_imu_train.reshape(-1, F)).reshape(N_train, T, F)
# Transform Test
N_test = X_imu_test.shape[0]
X_imu_test = scaler_imu.transform(X_imu_test.reshape(-1, F)).reshape(N_test, T, F)

# Scale HR
scaler_hr = StandardScaler()
X_hr_train = scaler_hr.fit_transform(X_hr_train)
X_hr_test = scaler_hr.transform(X_hr_test)

# ==========================================
# 4. Build Multi-Input Model
# ==========================================
def build_fused_model(window_size, n_channels):
    # --- Branch 1: IMU Data (CNN-BiLSTM) ---
    imu_input = layers.Input(shape=(window_size, n_channels), name="imu_input")

    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(imu_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)

    # --- Branch 2: Heart Rate Data (Dense) ---
    hr_input = layers.Input(shape=(1,), name="hr_input")
    y_branch = layers.Dense(16, activation="relu")(hr_input)

    # --- Fusion ---
    combined = layers.Concatenate()([x, y_branch])

    # --- Final Classifier ---
    z = layers.Dense(96, activation="relu")(combined)
    z = layers.Dropout(0.4)(z)
    outputs = layers.Dense(1, activation="sigmoid")(z)

    model = models.Model(inputs=[imu_input, hr_input], outputs=outputs)
    return model

model = build_fused_model(window_size=X_imu_train.shape[1], n_channels=X_imu_train.shape[2])
model.summary()

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# 5. Training
# ==========================================
checkpoint_path = f"{args.output_dir}/model_{args.mode}.h5"
csv_logger = callbacks.CSVLogger(f"{args.output_dir}/log_{args.mode}.csv")
early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy')

history = model.fit(
    x=[X_imu_train, X_hr_train], # Multi-Input List
    y=y_train,
    epochs=args.epochs,
    batch_size=args.batch_size,
    validation_data=([X_imu_test, X_hr_test], y_test),
    callbacks=[checkpoint, csv_logger, early_stop],
    verbose=2
)

# ==========================================
# 6. Evaluation
# ==========================================
print("\n--- Evaluation ---")
best_model = models.load_model(checkpoint_path)

# Predict (Pass list of inputs)
y_pred_prob = best_model.predict([X_imu_test, X_hr_test])
y_pred = (y_pred_prob > 0.5).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)

results = (
    f"Mode: {args.mode}\n"
    f"Accuracy:    {accuracy:.4f}\n"
    f"Sensitivity: {sensitivity:.4f}\n"
    f"Specificity: {specificity:.4f}\n"
    f"Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}\n"
)

print(results)
with open(f"{args.output_dir}/results_{args.mode}.txt", "w") as f:
    f.write(results)