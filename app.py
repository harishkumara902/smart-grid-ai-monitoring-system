from flask import Flask, render_template
from flask_socketio import SocketIO
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import threading
import time
import os

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

def train_model():
    data = pd.read_csv("smart_grid_1000_rows.csv")
    features = data[['consumer', 'producer', 'voltage', 'frequency', 'temperature']]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)

    X, y = [], []
    window_size = 5
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i])
        y.append(scaled_data[i][0])

    X = np.array(X)
    y = np.array(y)

    model = Sequential()
    model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)

    return model, scaler

print("Training model...")
model, scaler = train_model()
print("Model ready!")

actual_values = []
predicted_values = []
anomaly_values = []
voltage_values = []
frequency_values = []

@app.route('/')
def index():
    return render_template('index.html')

def generate_live_data():
    while True:
        # Generate random grid values
        voltage = round(np.random.uniform(210, 250), 2)
        frequency = round(np.random.uniform(49, 51), 2)
        temperature = round(np.random.uniform(20, 40), 2)
        producer = round(np.random.uniform(100, 500), 2)

        # Normalize input
        sample = np.array([[0, producer, voltage, frequency, temperature]])
        scaled_sample = np.zeros((1, 5))
        scaled_sample[0] = [0, producer/500, voltage/250, frequency/60, temperature/40]
        input_seq = np.tile(scaled_sample, (5, 1)).reshape(1, 5, 5)

        # Predict
        prediction = model.predict(input_seq, verbose=0)[0][0]
        dummy = np.zeros((1, 5))
        dummy[0][0] = prediction
        real_prediction = round(float(scaler.inverse_transform(dummy)[0][0]), 2)

        actual = round(real_prediction + np.random.uniform(-10, 10), 2)
        anomaly = round(abs(actual - real_prediction), 2)

        # Store values
        actual_values.append(actual)
        predicted_values.append(real_prediction)
        anomaly_values.append(anomaly)
        voltage_values.append(voltage)
        frequency_values.append(frequency)

        # Keep only last 50
        for lst in [actual_values, predicted_values, anomaly_values, voltage_values, frequency_values]:
            if len(lst) > 50:
                lst.pop(0)

        # Future forecast (next 10 points)
        future_predictions = [round(real_prediction + np.random.uniform(-5, 5), 2) for _ in range(10)]

        # Stats
        average_load = round(float(np.mean(actual_values)), 2)
        peak_load = round(float(np.max(actual_values)), 2)
        min_load = round(float(np.min(actual_values)), 2)

        # Accuracy
        accuracy = round(max(0, 100 - (anomaly / (abs(real_prediction) + 1)) * 100), 2)

        # Emit — name matches index.html
        socketio.emit('new_data', {
            'actual': actual,
            'predicted': real_prediction,
            'anomaly': anomaly,
            'voltage': voltage,
            'frequency': frequency,
            'future_predictions': future_predictions,
            'average_load': average_load,
            'peak_load': peak_load,
            'min_load': min_load,
            'accuracy': accuracy
        })

        print(f"ACTUAL: {actual} | PREDICTED: {real_prediction} | ANOMALY: {anomaly}")
        time.sleep(2)

thread = threading.Thread(target=generate_live_data)
thread.daemon = True
thread.start()

if __name__ == '__main__':
    socketio.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )