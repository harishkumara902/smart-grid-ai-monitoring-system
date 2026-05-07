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
socketio = SocketIO(app, cors_allowed_origins="*")

# Train model fresh on startup
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

# Store graph values
actual_values = []
predicted_values = []
anomaly_values = []

@app.route('/')
def index():
    return render_template(
        'index.html',
        actual=actual_values,
        predicted=predicted_values,
        anomaly=anomaly_values
    )

def generate_live_data():
    while True:
        voltage = np.random.uniform(210, 250)
        current = np.random.uniform(5, 20)
        power = voltage * current / 10

        sample = np.array([[voltage, current, power, 50, 25]])
        
        dummy_scaler = MinMaxScaler()
        dummy_scaler.fit(sample)
        
        scaled_sample = np.zeros((1, 5))
        scaled_sample[0] = [voltage/250, current/20, power/500, 50/60, 25/40]
        
        input_seq = np.tile(scaled_sample, (5, 1)).reshape(1, 5, 5)
        
        prediction = model.predict(input_seq, verbose=0)[0][0]
        
        dummy = np.zeros((1, 5))
        dummy[0][0] = prediction
        real_prediction = scaler.inverse_transform(dummy)[0][0]

        actual = real_prediction + np.random.uniform(-10, 10)
        anomaly = abs(actual - real_prediction)

        actual_values.append(round(actual, 2))
        predicted_values.append(round(real_prediction, 2))
        anomaly_values.append(round(anomaly, 2))

        if len(actual_values) > 30:
            actual_values.pop(0)
            predicted_values.pop(0)
            anomaly_values.pop(0)

        socketio.emit('newdata', {
            'actual': actual_values,
            'predicted': predicted_values,
            'anomaly': anomaly_values
        })

        print("ACTUAL:", actual)
        print("PREDICTED:", real_prediction)
        print("ANOMALY:", anomaly)

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