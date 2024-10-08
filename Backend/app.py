# backend/app.py

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from model_training import train_model  # Import the training function

app = Flask(__name__)

# Load your trained model
model = load_model('your_model.h5')  # Ensure your model is saved in the backend folder

# Load the dataset to access historical prices
df = pd.read_csv("path_to_your_bitcoin_data.csv")  # Update this path to your data

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    date_str = data['date']  # Get the date from the input

    # Convert string date to datetime object
    date = pd.to_datetime(date_str)

    # Retrieve the data for the date and the previous 59 days
    last_60_days = df.loc[(df['date'] <= date)].tail(60)  # Get last 60 days data before the date

    if len(last_60_days) < 60:
        return jsonify({'error': 'Not enough data for prediction'}), 400

    # Prepare the input for the model
    closing_prices = last_60_days['close'].values.reshape(-1, 1)  # Get closing prices
    scaler = MinMaxScaler()
    closing_prices_scaled = scaler.fit_transform(closing_prices)

    # Prepare input shape for LSTM (1, 60, 1)
    features = closing_prices_scaled.reshape(1, 60, 1)

    # Make prediction
    prediction = model.predict(features)

    # Inverse scale the prediction
    prediction_value = scaler.inverse_transform(prediction)[0][0]

    return jsonify({'prediction': prediction_value})

if __name__ == '__main__':
    app.run(debug=True)
