# Nova Fall Detection

Nova Fall Detection is a deep learning-based research project designed to detect fall events using multi-modal wearable sensor data. By fusing Inertial Measurement Unit (IMU) data with heart rate information, the project aims to improve the accuracy and reliability of fall detection algorithms for elderly monitoring and safety applications.

## Project Overview
The core objective of this project is to develop a robust fall detection system capable of distinguishing between actual falls and Activities of Daily Living (ADL). The system utilizes a multi-input deep learning architecture to process both motion sequences and physiological signals (heart rate).

## Model Architecture
The project employs a **Hybrid CNN-BiLSTM Multi-Input Model** developed using TensorFlow/Keras.

### 1. IMU Branch (Spatial-Temporal Extraction)
*   **1D-CNN Layers**: Two layers of 1D Convolutional Neural Networks (128 and 256 filters) are used to extract local spatial features from the accelerometer, gyroscope, and magnetometer data.
*   **Batch Normalization & Pooling**: Used to stabilize training and reduce dimensionality.
*   **Stacked BiLSTM Layers**: Two Bidirectional LSTM layers (128 units each) capture the temporal dependencies and long-term patterns in the motion data.

### 2. Heart Rate Branch
*   **Dense Layer**: A dedicated branch for processing heart rate data, allowing the model to learn physiological markers associated with fall events.

### 3. Data Fusion & Classification
*   **Concatenation**: Features from both the motion (CNN-BiLSTM) and heart rate branches are fused together.
*   **Classifier**: Dense layers with Dropout (0.4) lead to a final Sigmoid output, providing a probability for the "Fall" vs "No Fall" binary classification.

## Experimentation
The project includes a comprehensive suite of experiments to evaluate the impact of different sensor configurations and the inclusion of heart rate data.

### Sensor Configurations
Experiments are conducted across three primary modes:
*   **3-Axis**: Accelerometer only + Heart Rate.
*   **6-Axis**: Accelerometer + Gyroscope + Heart Rate.
*   **9-Axis**: Accelerometer + Gyroscope + Magnetometer + Heart Rate.

### Comparative Study
A side-by-side comparison is performed with a **Non-HR (No Heart Rate)** version of the model to quantify the performance gain provided by physiological data.

### Evaluation Metrics
The models are evaluated based on:
*   **Accuracy**: Overall detection performance.
*   **Sensitivity (Recall)**: Ability to correctly identify all fall events.
*   **Specificity**: Ability to avoid false alarms during normal daily activities.

## Usage
### 1. Preprocessing
Prepare the raw sensor data into processed tensors:
```bash
./run_prep.sh
```

### 2. Hyperparameter Tuning (Optional)
Optimize the model architecture:
```bash
./run_tuning.sh
```

### 3. Run Main Experiments
Execute the full training and evaluation suite:
```bash
./run_experiment.sh
```

## Dataset

The project utilizes the **BITS-2 Dataset for Fall Detection**, a comprehensive collection of sensor data specifically designed for fall detection research.

*   **Dataset Name**: BITS-2 Dataset for Fall detection
*   **Source**: [Zenodo](https://doi.org/10.5281/zenodo.8082667)
*   **Citation**: Purab Nandi, & K.R. Anupama. (2023). BITS-2 Dataset for Fall detection [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8082667
