import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# LOAD TRAINED MODEL
model = load_model("smart_grid_model.h5", compile=False)

# LOAD CSV
data = pd.read_csv("smart_grid_1000_rows.csv")

# SELECT FEATURES
features = data[['consumer', 'producer', 'voltage', 'frequency', 'temperature']]

# NORMALIZE DATA
scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(features)

# TAKE LAST 5 ROWS
last_sequence = scaled_data[-5:]

# RESHAPE FOR LSTM
last_sequence = np.reshape(last_sequence, (1, 5, 5))

# PREDICT
prediction = model.predict(last_sequence)

print("\nSCALED PREDICTION:")
print(prediction)

# CONVERT BACK TO ORIGINAL VALUE

dummy_array = np.zeros((1, 5))

dummy_array[0][0] = prediction[0][0]

real_prediction = scaler.inverse_transform(dummy_array)

print("\nPREDICTED FUTURE CONSUMER LOAD:")
print(real_prediction[0][0])