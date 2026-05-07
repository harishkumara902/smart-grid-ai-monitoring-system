from flask import Flask, render_template
from flask_socketio import SocketIO
import numpy as np
import tensorflow as tf
import threading
import time
import os

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load trained model
model = tf.keras.models.load_model(
    "smart_grid_model.h5",
    compile=False
)

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

        # Generate random smart grid data
        voltage = np.random.uniform(210, 250)
        current = np.random.uniform(5, 20)
        power = voltage * current / 10

        # Prepare input for model
        sample = np.array([[voltage, current, power]])

        # Predict
        prediction = model.predict(sample, verbose=0)[0][0]

        # Simulated actual value
        actual = prediction + np.random.uniform(-10, 10)

        # Anomaly detection
        anomaly = abs(actual - prediction)

        # Store values
        actual_values.append(round(actual, 2))
        predicted_values.append(round(prediction, 2))
        anomaly_values.append(round(anomaly, 2))

        # Keep only latest 30 values
        if len(actual_values) > 30:
            actual_values.pop(0)
            predicted_values.pop(0)
            anomaly_values.pop(0)

        # Send data to frontend
        socketio.emit('newdata', {
            'actual': actual_values,
            'predicted': predicted_values,
            'anomaly': anomaly_values
        })

        print("ACTUAL:", actual)
        print("PREDICTED:", prediction)
        print("ANOMALY:", anomaly)

        time.sleep(2)

# Start background thread
thread = threading.Thread(target=generate_live_data)
thread.daemon = True
thread.start()

if __name__ == '__main__':
    socketio.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )