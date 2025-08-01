import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Load the trained model and scaler
model = load_model("earthquake_lstm_model_v2.h5")
scaler = joblib.load("scaler.save")

# Input: Last 14 days of data as a DataFrame
def predict_earthquake_next_day(last_14_days_df):
    # Ensure column names match
    required_features = ['longitude', 'latitude', 'depth', 'significance', 'tsunami']
    
    if not all(col in last_14_days_df.columns for col in required_features):
        raise ValueError("Input must contain the required columns:", required_features)

    # Use only the last 14 rows (just in case)
    last_14_days_df = last_14_days_df.sort_values('date').tail(14)

    # Normalize features
    scaled = scaler.transform(last_14_days_df[required_features])

    # Reshape for LSTM input: (1 sample, 14 timesteps, 5 features)
    input_seq = np.expand_dims(scaled, axis=0)

    # Predict
    prob = model.predict(input_seq)[0][0]
    label = int(prob > 0.5)

    print(f"Predicted Probability: {prob:.4f}")
    return "YES - Earthquake likely" if label == 1 else "NO - Earthquake unlikely"

recent_df = pd.read_csv("last_14_days_high_risk_with_tsunami.csv")  # Must have the required columns
prediction = predict_earthquake_next_day(recent_df)
print("Prediction for tomorrow:", prediction)