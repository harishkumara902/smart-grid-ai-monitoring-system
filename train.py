import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# LOAD CSV
data = pd.read_csv("smart_grid_1000_rows.csv")

# SELECT FEATURES
features = data[['consumer', 'producer', 'voltage', 'frequency', 'temperature']]

# NORMALIZATION
scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(features)

# CREATE SEQUENCES
X = []
y = []

window_size = 5

for i in range(window_size, len(scaled_data)):

    X.append(scaled_data[i-window_size:i])

    # Future consumer value
    y.append(scaled_data[i][0])

# CONVERT TO NUMPY
X = np.array(X)
y = np.array(y)

print("X SHAPE:", X.shape)
print("Y SHAPE:", y.shape)

# BUILD MODEL
model = Sequential()

model.add(
    LSTM(
        64,
        input_shape=(X.shape[1], X.shape[2])
    )
)

model.add(Dense(1))

# COMPILE MODEL
model.compile(
    optimizer='adam',
    loss='mse'
)

# TRAIN MODEL
model.fit(
    X,
    y,
    epochs=20,
    batch_size=32
)

# SAVE MODEL
model.save("smart_grid_model.h5")

print("\nMODEL TRAINED SUCCESSFULLY")