from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import torch.nn as nn
import torch
import pickle
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Initialize Flask App
app = Flask(__name__)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ====== Load Models Once at Startup ======
logging.basicConfig(level=logging.INFO)

try:
    # Load models only once
    custom_objects = {"mse": MeanSquaredError()}
    
    # Cyclone Models
    lstm_model = load_model("cyclone_lstm_model.h5", custom_objects=custom_objects)
    speed_model = joblib.load("speed_model.pkl")
    dir_model = joblib.load("dir_model.pkl")
    
    # Load Scalers
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    
    # Severity Classification Models
    encoder = load_model("severity_encoder.h5")  
    scaler_severity = joblib.load("severity_scaler.pkl")
    kmeans = joblib.load("severity_kmeans.pkl")

    # Load Earthquake Models
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('knn_model.pkl', 'rb') as f:
        kmeans_eq = pickle.load(f)

    autoencoder = Autoencoder(input_dim=4, hidden_dim=16, latent_dim=8)
    autoencoder.load_state_dict(torch.load("autoencoder2.pth", map_location=torch.device('cpu')))
    autoencoder.eval()  # Put model in evaluation mode
 
    # Load PyTorch Autoencoder Correctly
 # Set PyTorch model to evaluation mode

    # Severity Labels
    severity_labels = {0: "Mild", 1: "Moderate", 2: "Severe", 3: "Catastrophic"}
    
    logging.info(" All models loaded successfully!")

except Exception as e:
    logging.error(f"🚨 Error loading models: {e}")

# ====== Flask Routes ======

@app.route('/')
def home():
    return "Disaster Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict_earthquake():
    try:
        data = request.json
        input_data = np.array([[data['magnitude'], data['depth'], data['latitude'], data['longitude']]])
        input_scaled = scaler.transform(input_data)
        input_tensor = torch.FloatTensor(input_scaled)

        with torch.no_grad():
            encoded_features = autoencoder.encoder(input_tensor).detach().numpy()
            cluster_label = kmeans_eq.predict(encoded_features)[0]
            reconstructed_data = autoencoder(input_tensor)
            reconstruction_error = torch.mean((input_tensor - reconstructed_data) ** 2).item()

        anomaly = reconstruction_error > 0.01  # Threshold for anomaly detection

        return jsonify({
            "cluster": int(cluster_label),
            "reconstruction_error": reconstruction_error,
            "anomaly": bool(anomaly)
        })
    
    except Exception as e:
        logging.error(f"Error in predict_earthquake: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/predict-speed', methods=['POST'])
def predict_speed():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        # Convert ISO_TIME to datetime
        df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
        df['HOUR'] = df['ISO_TIME'].dt.hour
        df['MONTH'] = df['ISO_TIME'].dt.month

        # Compute directional encodings
        df['dir_sin'] = np.sin(np.deg2rad(df['STORM_DIR']))
        df['dir_cos'] = np.cos(np.deg2rad(df['STORM_DIR']))

        # Interaction Features
        df['lat_lon_interaction'] = df['LAT'] * df['LON']
        df['speed_lat_interaction'] = df['STORM_SPEED'] * df['LAT']
        df['speed_lon_interaction'] = df['STORM_SPEED'] * df['LON']

        # ✅ Restore missing lag & moving average features
        df['STORM_SPEED_LAG1'] = df['STORM_SPEED']
        df['LAT_LAG'] = df['LAT']
        df['LON_LAG'] = df['LON']
        df['SPEED_MA3'] = df['STORM_SPEED']

        # ✅ Ensure feature list matches model expectations
        features = [
            'LAT', 'LON', 'STORM_SPEED', 'HOUR', 'MONTH', 
            'dir_sin', 'dir_cos', 'STORM_SPEED_LAG1', 'LAT_LAG', 'LON_LAG', 'SPEED_MA3',
            'lat_lon_interaction', 'speed_lat_interaction'
        ]

        # Check if model expects a different number of features
        if len(features) != speed_model.get_booster().num_features():
            return jsonify({'error': f'Feature shape mismatch, expected {speed_model.get_booster().num_features()}, got {len(features)}'}), 400

        # Prepare input for prediction
        X = df[features].values

        # Predict Speed & Direction
        speed_pred = speed_model.predict(X).tolist()
        dir_pred = dir_model.predict(X).tolist()

        return jsonify({'predicted_speed': speed_pred, 'predicted_direction': dir_pred})

    except Exception as e:
        logging.error(f"Error in predict_speed: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/classify-severity', methods=['POST'])
def classify_severity():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
        df['HOUR'] = df['ISO_TIME'].dt.hour.fillna(0)
        df['MONTH'] = df['ISO_TIME'].dt.month.fillna(0)
        df['dir_sin'] = np.sin(np.deg2rad(df['STORM_DIR']))
        df['dir_cos'] = np.cos(np.deg2rad(df['STORM_DIR']))

        features = ['LAT', 'LON', 'STORM_SPEED', 'HOUR', 'MONTH', 'dir_sin', 'dir_cos']
        X_scaled = scaler_severity.transform(df[features].values)

        latent_features = encoder.predict(X_scaled)
        cluster_labels = kmeans.predict(latent_features)
        df['Severity'] = [severity_labels.get(c, "Unknown") for c in cluster_labels]

        return jsonify(df[['LAT', 'LON', 'STORM_SPEED', 'Severity']].to_dict(orient='records'))

    except Exception as e:
        logging.error(f"Error in classify_severity: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# ====== Run the Flask App ======
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=False, host='0.0.0.0', port=port)
