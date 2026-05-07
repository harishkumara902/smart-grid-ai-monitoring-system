import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# CREATE TEST SEQUENCES
X = []
actual_values = []

window_size = 5

for i in range(window_size, len(scaled_data)):

    X.append(scaled_data[i-window_size:i])

    actual_values.append(features.iloc[i]['consumer'])

X = np.array(X)

# PREDICT
predictions = model.predict(X)

# CONVERT PREDICTIONS BACK
predicted_values = []

for pred in predictions:

    dummy = np.zeros((1,5))

    dummy[0][0] = pred[0]

    real_value = scaler.inverse_transform(dummy)

    predicted_values.append(real_value[0][0])

# PLOT GRAPH
plt.figure(figsize=(12,6))

plt.plot(actual_values, label="Actual Consumer Load")

plt.plot(predicted_values, label="Predicted Consumer Load")

plt.xlabel("Time")

plt.ylabel("Consumer Load")

plt.title("Smart Grid Load Prediction")

plt.legend()

plt.show()