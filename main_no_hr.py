import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='6axis', choices=['3axis', '6axis', '9axis'])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--data_dir', type=str, default='./processed_tensors_no_hr')
parser.add_argument('--output_dir', type=str, default='./results_no_hr')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# 1. Load Data
data_path = os.path.join(args.data_dir, f"data_{args.mode}.npz")
data = np.load(data_path)
X_imu = data['X_imu']
y = data['y']

X_imu_train, X_imu_test, y_train, y_test = train_test_split(
    X_imu, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Scaling
scaler_imu = StandardScaler()
N_train, T, F = X_imu_train.shape
X_imu_train = scaler_imu.fit_transform(X_imu_train.reshape(-1, F)).reshape(N_train, T, F)
N_test = X_imu_test.shape[0]
X_imu_test = scaler_imu.transform(X_imu_test.reshape(-1, F)).reshape(N_test, T, F)

# 3. Model Architecture (NO HR BRANCH)
def build_imu_only_model(window_size, n_channels):
    imu_input = layers.Input(shape=(window_size, n_channels), name="imu_input")

    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(imu_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)

    z = layers.Dense(64, activation="relu")(x)
    z = layers.Dropout(0.5)(z)
    outputs = layers.Dense(1, activation="sigmoid")(z)

    return models.Model(inputs=imu_input, outputs=outputs)

model = build_imu_only_model(window_size=X_imu_train.shape[1], n_channels=X_imu_train.shape[2])

model.compile(
    optimizer=optimizers.Adam(learning_rate=5e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 4. Train
checkpoint_path = f"{args.output_dir}/model_{args.mode}.h5"
csv_logger = callbacks.CSVLogger(f"{args.output_dir}/log_{args.mode}.csv")
early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy')

history = model.fit(
    x=X_imu_train,
    y=y_train,
    epochs=args.epochs,
    batch_size=args.batch_size,
    validation_data=(X_imu_test, y_test),
    callbacks=[checkpoint, csv_logger, early_stop],
    verbose=2
)

# 5. Evaluate
print("\n--- Evaluation ---")
try:
    y_pred_prob = model.predict(X_imu_test, batch_size=args.batch_size)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = y_test.flatten()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    results = (
        f"Mode: {args.mode} (NO HR)\n"
        f"Accuracy:    {accuracy:.4f}\n"
        f"Sensitivity: {sensitivity:.4f}\n"
        f"Specificity: {specificity:.4f}\n"
        f"Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}\n"
    )

    print(results)
    with open(f"{args.output_dir}/results_{args.mode}.txt", "w") as f:
        f.write(results)
except Exception as e:
    print(f"\nCRITICAL ERROR DURING EVALUATION: {e}\n")