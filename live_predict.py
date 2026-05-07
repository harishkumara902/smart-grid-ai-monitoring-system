import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# LOAD MODEL
model = load_model("smart_grid_model.h5", compile=False)

# LOAD DATA
data = pd.read_csv("smart_grid_1000_rows.csv")

# SELECT FEATURES
features = data[['consumer', 'producer', 'voltage', 'frequency', 'temperature']]

# SCALE DATA
scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(features)

# INITIAL LIVE BUFFER
window_size = 5

live_buffer = list(scaled_data[:window_size])

print("LIVE PREDICTION STARTED...\n")

# STREAM DATA
for i in range(window_size, len(scaled_data)):

    # NEW LIVE DATA
    new_data = scaled_data[i]

    # REMOVE OLD DATA
    live_buffer.pop(0)

    # ADD NEW DATA
    live_buffer.append(new_data)

    # CONVERT TO ARRAY
    input_data = np.array(live_buffer)

    # RESHAPE FOR LSTM
    input_data = np.reshape(input_data, (1, window_size, 5))

    # PREDICT
    prediction = model.predict(input_data, verbose=0)

    # INVERSE SCALE
    dummy = np.zeros((1,5))

    dummy[0][0] = prediction[0][0]

    real_prediction = scaler.inverse_transform(dummy)

    print("Predicted Consumer Load:", real_prediction[0][0])

    # LIVE DELAY
    time.sleep(1)