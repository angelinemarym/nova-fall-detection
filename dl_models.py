import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_only_model(window_size, n_channels, use_hr=False):
    imu_input = layers.Input(shape=(window_size, n_channels), name="imu_input")
    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(imu_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    if use_hr:
        hr_input = layers.Input(shape=(1,), name="hr_input")
        hr_branch = layers.Dense(16, activation="relu")(hr_input)
        combined = layers.Concatenate()([x, hr_branch])
        inputs = [imu_input, hr_input]
    else:
        combined = x
        inputs = imu_input

    z = layers.Dense(96, activation="relu")(combined)
    z = layers.Dropout(0.4)(z)
    outputs = layers.Dense(1, activation="sigmoid")(z)
    return models.Model(inputs=inputs, outputs=outputs, name="CNN_Only")

def build_lstm_only_model(window_size, n_channels, use_hr=False):
    imu_input = layers.Input(shape=(window_size, n_channels), name="imu_input")
    x = layers.LSTM(64, return_sequences=True)(imu_input)
    x = layers.LSTM(64, return_sequences=False)(x)

    if use_hr:
        hr_input = layers.Input(shape=(1,), name="hr_input")
        hr_branch = layers.Dense(16, activation="relu")(hr_input)
        combined = layers.Concatenate()([x, hr_branch])
        inputs = [imu_input, hr_input]
    else:
        combined = x
        inputs = imu_input

    z = layers.Dense(96, activation="relu")(combined)
    z = layers.Dropout(0.4)(z)
    outputs = layers.Dense(1, activation="sigmoid")(z)
    return models.Model(inputs=inputs, outputs=outputs, name="LSTM_Only")

def build_bilstm_only_model(window_size, n_channels, use_hr=False):
    imu_input = layers.Input(shape=(window_size, n_channels), name="imu_input")
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(imu_input)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)

    if use_hr:
        hr_input = layers.Input(shape=(1,), name="hr_input")
        hr_branch = layers.Dense(16, activation="relu")(hr_input)
        combined = layers.Concatenate()([x, hr_branch])
        inputs = [imu_input, hr_input]
    else:
        combined = x
        inputs = imu_input

    z = layers.Dense(96, activation="relu")(combined)
    z = layers.Dropout(0.4)(z)
    outputs = layers.Dense(1, activation="sigmoid")(z)
    return models.Model(inputs=inputs, outputs=outputs, name="BiLSTM_Only")

def build_unidirectional_cnn_lstm_model(window_size, n_channels, use_hr=False):
    imu_input = layers.Input(shape=(window_size, n_channels), name="imu_input")
    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(imu_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=False)(x)

    if use_hr:
        hr_input = layers.Input(shape=(1,), name="hr_input")
        hr_branch = layers.Dense(16, activation="relu")(hr_input)
        combined = layers.Concatenate()([x, hr_branch])
        inputs = [imu_input, hr_input]
    else:
        combined = x
        inputs = imu_input

    z = layers.Dense(96, activation="relu")(combined)
    z = layers.Dropout(0.4)(z)
    outputs = layers.Dense(1, activation="sigmoid")(z)
    return models.Model(inputs=inputs, outputs=outputs, name="CNN_LSTM_Unidirectional")

def build_transformer_model(window_size, n_channels, use_hr=False):
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

    imu_input = layers.Input(shape=(window_size, n_channels), name="imu_input")
    x = transformer_encoder(imu_input, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    x = layers.GlobalAveragePooling1D()(x)

    if use_hr:
        hr_input = layers.Input(shape=(1,), name="hr_input")
        hr_branch = layers.Dense(16, activation="relu")(hr_input)
        combined = layers.Concatenate()([x, hr_branch])
        inputs = [imu_input, hr_input]
    else:
        combined = x
        inputs = imu_input

    z = layers.Dense(96, activation="relu")(combined)
    z = layers.Dropout(0.4)(z)
    outputs = layers.Dense(1, activation="sigmoid")(z)
    return models.Model(inputs=inputs, outputs=outputs, name="Transformer")

def build_cnn_bilstm_model(window_size, n_channels, use_hr=False):
    imu_input = layers.Input(shape=(window_size, n_channels), name="imu_input")
    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(imu_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)

    if use_hr:
        hr_input = layers.Input(shape=(1,), name="hr_input")
        hr_branch = layers.Dense(16, activation="relu")(hr_input)
        combined = layers.Concatenate()([x, hr_branch])
        inputs = [imu_input, hr_input]
    else:
        combined = x
        inputs = imu_input

    z = layers.Dense(96, activation="relu")(combined)
    z = layers.Dropout(0.4)(z)
    outputs = layers.Dense(1, activation="sigmoid")(z)
    return models.Model(inputs=inputs, outputs=outputs, name="CNN_BiLSTM")

def build_cnn_gru_model(window_size, n_channels, use_hr=False):
    imu_input = layers.Input(shape=(window_size, n_channels), name="imu_input")
    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(imu_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.GRU(64, return_sequences=True)(x)
    x = layers.GRU(64, return_sequences=False)(x)

    if use_hr:
        hr_input = layers.Input(shape=(1,), name="hr_input")
        hr_branch = layers.Dense(16, activation="relu")(hr_input)
        combined = layers.Concatenate()([x, hr_branch])
        inputs = [imu_input, hr_input]
    else:
        combined = x
        inputs = imu_input

    z = layers.Dense(96, activation="relu")(combined)
    z = layers.Dropout(0.4)(z)
    outputs = layers.Dense(1, activation="sigmoid")(z)
    return models.Model(inputs=inputs, outputs=outputs, name="CNN_GRU")

def build_cnn_bigru_model(window_size, n_channels, use_hr=False):
    imu_input = layers.Input(shape=(window_size, n_channels), name="imu_input")
    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(imu_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=False))(x)

    if use_hr:
        hr_input = layers.Input(shape=(1,), name="hr_input")
        hr_branch = layers.Dense(16, activation="relu")(hr_input)
        combined = layers.Concatenate()([x, hr_branch])
        inputs = [imu_input, hr_input]
    else:
        combined = x
        inputs = imu_input

    z = layers.Dense(96, activation="relu")(combined)
    z = layers.Dropout(0.4)(z)
    outputs = layers.Dense(1, activation="sigmoid")(z)
    return models.Model(inputs=inputs, outputs=outputs, name="CNN_BiGRU")

def build_gru_only_model(window_size, n_channels, use_hr=False):
    imu_input = layers.Input(shape=(window_size, n_channels), name="imu_input")
    x = layers.GRU(64, return_sequences=True)(imu_input)
    x = layers.GRU(64, return_sequences=False)(x)

    if use_hr:
        hr_input = layers.Input(shape=(1,), name="hr_input")
        hr_branch = layers.Dense(16, activation="relu")(hr_input)
        combined = layers.Concatenate()([x, hr_branch])
        inputs = [imu_input, hr_input]
    else:
        combined = x
        inputs = imu_input

    z = layers.Dense(96, activation="relu")(combined)
    z = layers.Dropout(0.4)(z)
    outputs = layers.Dense(1, activation="sigmoid")(z)
    return models.Model(inputs=inputs, outputs=outputs, name="GRU_Only")

def build_bigru_only_model(window_size, n_channels, use_hr=False):
    imu_input = layers.Input(shape=(window_size, n_channels), name="imu_input")
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(imu_input)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=False))(x)

    if use_hr:
        hr_input = layers.Input(shape=(1,), name="hr_input")
        hr_branch = layers.Dense(16, activation="relu")(hr_input)
        combined = layers.Concatenate()([x, hr_branch])
        inputs = [imu_input, hr_input]
    else:
        combined = x
        inputs = imu_input

    z = layers.Dense(96, activation="relu")(combined)
    z = layers.Dropout(0.4)(z)
    outputs = layers.Dense(1, activation="sigmoid")(z)
    return models.Model(inputs=inputs, outputs=outputs, name="BiGRU_Only")

def _tcn_residual_block(x, filters, kernel_size, dilation_rate, dropout=0.2):
    residual = x
    x = layers.Conv1D(filters, kernel_size, padding='causal',
                      dilation_rate=dilation_rate, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters, kernel_size, padding='causal',
                      dilation_rate=dilation_rate, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    if residual.shape[-1] != filters:
        residual = layers.Conv1D(filters, 1, padding='same')(residual)
    return layers.Add()([x, residual])

def build_tcn_model(window_size, n_channels, use_hr=False):
    imu_input = layers.Input(shape=(window_size, n_channels), name="imu_input")
    x = imu_input
    for dilation in [1, 2, 4, 8]:
        x = _tcn_residual_block(x, filters=128, kernel_size=3, dilation_rate=dilation, dropout=0.2)
    x = layers.GlobalAveragePooling1D()(x)

    if use_hr:
        hr_input = layers.Input(shape=(1,), name="hr_input")
        hr_branch = layers.Dense(16, activation="relu")(hr_input)
        combined = layers.Concatenate()([x, hr_branch])
        inputs = [imu_input, hr_input]
    else:
        combined = x
        inputs = imu_input

    z = layers.Dense(96, activation="relu")(combined)
    z = layers.Dropout(0.3)(z)
    outputs = layers.Dense(1, activation="sigmoid")(z)
    return models.Model(inputs=inputs, outputs=outputs, name="TCN")

def _se_block(x, ratio=16):
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(max(filters // ratio, 1), activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, filters))(se)
    return layers.Multiply()([x, se])

def build_multiscale_se_bilstm_model(window_size, n_channels, use_hr=False):
    imu_input = layers.Input(shape=(window_size, n_channels), name="imu_input")
    b3  = layers.Conv1D(64,  3,  padding='same', activation='relu')(imu_input)
    b7  = layers.Conv1D(64,  7,  padding='same', activation='relu')(imu_input)
    b15 = layers.Conv1D(64, 15,  padding='same', activation='relu')(imu_input)
    x = layers.Concatenate()([b3, b7, b15])
    x = layers.BatchNormalization()(x)
    x = _se_block(x, ratio=16)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = _se_block(x, ratio=16)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)

    if use_hr:
        hr_input = layers.Input(shape=(1,), name="hr_input")
        hr_branch = layers.Dense(16, activation="relu")(hr_input)
        combined = layers.Concatenate()([x, hr_branch])
        inputs = [imu_input, hr_input]
    else:
        combined = x
        inputs = imu_input

    z = layers.Dense(96, activation="relu")(combined)
    z = layers.Dropout(0.4)(z)
    outputs = layers.Dense(1, activation="sigmoid")(z)
    return models.Model(inputs=inputs, outputs=outputs, name="MultiScale_SE_BiLSTM")
