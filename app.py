from flask import Flask, render_template
from flask_socketio import SocketIO
import pandas as pd
import numpy as np
import threading
import time
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# FLASK APP
app = Flask(__name__)

# SOCKETIO
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading'
)

# LOAD MODEL
model = load_model(
    "smart_grid_model.h5",
    compile=False
)

# LOAD DATA
data = pd.read_csv(
    "smart_grid_1000_rows.csv"
)

# SELECT FEATURES
features = data[
    [
        'consumer',
        'producer',
        'voltage',
        'frequency',
        'temperature'
    ]
]

# SCALE DATA
scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(
    features
)

window_size = 5

# ANALYTICS VARIABLES
total_load = 0
reading_count = 0
peak_load = 0
min_load = 999999
anomaly_count = 0

# HOME PAGE
@app.route('/')
def index():

    return render_template(
        'index.html'
    )

# LIVE DATA LOOP
def send_live_data():

    global total_load
    global reading_count
    global peak_load
    global min_load
    global anomaly_count

    live_buffer = list(
        scaled_data[:window_size]
    )

    i = window_size

    while True:

        try:

            # NEW DATA
            new_data = scaled_data[i]

            # SLIDING WINDOW
            live_buffer.pop(0)

            live_buffer.append(new_data)

            # PREPARE INPUT
            input_data = np.array(
                live_buffer
            )

            input_data = np.reshape(
                input_data,
                (1, window_size, 5)
            )

            # FUTURE FORECASTING
            future_predictions = []

            temp_buffer = live_buffer.copy()

            for _ in range(10):

                temp_input = np.array(
                    temp_buffer
                )

                temp_input = np.reshape(
                    temp_input,
                    (1, window_size, 5)
                )

                future_pred = model.predict(
                    temp_input,
                    verbose=0
                )

                dummy_future = np.zeros((1, 5))

                dummy_future[0][0] = future_pred[0][0]

                real_future = scaler.inverse_transform(
                    dummy_future
                )

                future_value = float(
                    real_future[0][0]
                )

                future_predictions.append(
                    future_value
                )

                next_row = temp_buffer[-1].copy()

                next_row[0] = future_pred[0][0]

                temp_buffer.pop(0)

                temp_buffer.append(next_row)

            # CURRENT PREDICTION
            prediction = model.predict(
                input_data,
                verbose=0
            )

            # INVERSE SCALE
            dummy = np.zeros((1, 5))

            dummy[0][0] = prediction[0][0]

            real_prediction = scaler.inverse_transform(
                dummy
            )

            predicted_value = float(
                real_prediction[0][0]
            )

            actual_value = float(
                features.iloc[i]['consumer']
            )

            voltage_value = float(
                features.iloc[i]['voltage']
            )

            frequency_value = float(
                features.iloc[i]['frequency']
            )

            producer_value = float(
                features.iloc[i]['producer']
            )

            temperature_value = float(
                features.iloc[i]['temperature']
            )

            # ANALYTICS
            total_load += actual_value

            reading_count += 1

            average_load = total_load / reading_count

            if actual_value > peak_load:
                peak_load = actual_value

            if actual_value < min_load:
                min_load = actual_value

            # ANOMALY SCORE
            anomaly_score = abs(
                actual_value - predicted_value
            )

            if anomaly_score > 10:
                anomaly_count += 1

            # ACCURACY
            accuracy = max(
                0,
                100 - (
                    anomaly_score /
                    actual_value * 100
                )
            )

            print("ACTUAL:", actual_value)
            print("PREDICTED:", predicted_value)
            print("ANOMALY:", anomaly_score)

            # SEND TO DASHBOARD
            socketio.emit(

                'new_data',

                {

                    'actual': actual_value,

                    'predicted': predicted_value,

                    'voltage': voltage_value,

                    'frequency': frequency_value,

                    'producer': producer_value,

                    'temperature': temperature_value,

                    'anomaly': anomaly_score,

                    'average_load': average_load,

                    'peak_load': peak_load,

                    'min_load': min_load,

                    'anomaly_count': anomaly_count,

                    'accuracy': accuracy,

                    'future_predictions': future_predictions
                }
            )

            # NEXT DATA
            i += 1

            if i >= len(scaled_data):

                i = window_size

            # LIVE DELAY
            time.sleep(1)

        except Exception as e:

            print("ERROR:", e)

# MAIN
if __name__ == '__main__':

    thread = threading.Thread(
        target=send_live_data
    )

    thread.daemon = True

    thread.start()

    socketio.run(

        app,

        host='0.0.0.0',

        port=int(os.environ.get("PORT", 5000)),

        debug=True,

        allow_unsafe_werkzeug=True
    )