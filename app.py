from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ---------------------- MODEL LOADING ----------------------

try:
    heatwave_model = joblib.load('ml/rfc_heatwave_model.pkl')
except Exception as e:
    heatwave_model = None
    print(f"Heatwave Model loading error: {e}")

try:
    flood_model = joblib.load("ml/flood_prediction_model.pkl")
except Exception as e:
    flood_model = None
    print(f"Flood Model loading error: {e}")

try:
    earthquake_model = load_model("ml/earthquake_lstm_model_v2.h5")
    earthquake_scaler = joblib.load("ml/scaler.save")
except Exception as e:
    print("Error loading earthquake model:", e)
    earthquake_model = None
    earthquake_scaler = None

try:
    thunderstorm_model = joblib.load('ml/thunderstorm_classifier.pkl')
except Exception as e:
    thunderstorm_model = None
    print(f"Thunderstorm Model loading error: {e}")

try:
        # Load the trained model
    forest_fire_model = joblib.load('ml/forest_fire_model.pkl')

    # Load the scaler used during training
    fire_scaler = joblib.load('ml/fire_scaler.pkl')
except Exception as e:
    prediction=None

try:
    cyclone_model = joblib.load("ml/cyclone_model.pkl")
except Exception as e:
    print("Cyclone model load error:", e)
    cyclone_model = None



# ---------------------- ROUTES ----------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/alerts')
def alerts():
    return render_template('pages/alerts.html')

@app.route('/analysis')
def analysis():
    return render_template('pages/analysis.html')

@app.route('/cyclone')
def cyclone():
    return render_template('pages/cyclone.html')

@app.route("/predict-cyclone", methods=["POST"])
def predict_cyclone():
    try:
        data = request.get_json()

        required_features = ['lat', 'long', 'wind', 'pressure', 'tropicalstorm_force_diameter', 'hurricane_force_diameter']
        features = [float(data.get(feat, 0.0)) for feat in required_features]
        input_array = np.array([features])

        if cyclone_model:
            prediction = cyclone_model.predict(input_array)[0]
            result = "🌀 Cyclone Likely" if prediction == 1 else "✅ No Cyclone Risk"
        else:
            result = "❌ Cyclone model not loaded"

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'result': f"❌ Error: {str(e)}"})


@app.route('/earthquake')
def earthquake():
    return render_template('pages/earthquake.html')

@app.route("/predict-earthquake", methods=["POST"])
def predict_earthquake():
    try:
        if 'timeSeriesFile' not in request.files:
            return jsonify({"result": "❌ No CSV file uploaded"})

        file = request.files['timeSeriesFile']
        if file.filename == '':
            return jsonify({"result": "❌ No file selected"})

        df = pd.read_csv(file)
        required_columns = ['longitude', 'latitude', 'depth', 'significance', 'tsunami', 'date']
        if not all(col in df.columns for col in required_columns):
            return jsonify({"result": f"❌ CSV must contain columns: {', '.join(required_columns)}"})

        df = df.sort_values('date').tail(14)
        scaled = earthquake_scaler.transform(df[['longitude', 'latitude', 'depth', 'significance', 'tsunami']])
        input_seq = np.expand_dims(scaled, axis=0)

        prob = earthquake_model.predict(input_seq)[0][0]
        label = int(prob > 0.5)

        result = f"{'🌍 Earthquake Likely' if label else '✅ No Earthquake Risk'} (Confidence: {prob:.2%})"
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"result": f"Error: {str(e)}"})

@app.route('/fire')
def fire():
    return render_template('pages/fire.html')

@app.route('/predict-fire', methods=['POST'])
def predict_fire():
    try:
        data = request.get_json()

        # Only use the features used during model training
        temperature = float(data.get('temperature'))
        wind = float(data.get('wind'))
        isi = float(data.get('isi'))

        features = [[temperature, wind, isi]]  # Wrap inside list for 2D shape

        # Apply scaling if your model needs scaled input
        if fire_scaler:
            features = fire_scaler.transform(features)

        prediction = forest_fire_model.predict(features)[0]

        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        print("🔥 Error during fire prediction:", e)
        return jsonify({'error': str(e)}), 500




@app.route('/flood')
def flood():
    return render_template('pages/flood.html')

@app.route("/predict-flood", methods=["POST"])
def predict_flood():
    try:
        feature_order = [
            'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
            'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation', 'AgriculturalPractices',
            'Encroachments', 'IneffectiveDisasterPreparedness', 'DrainageSystems', 'CoastalVulnerability',
            'Landslides', 'Watersheds', 'DeterioratingInfrastructure', 'PopulationScore',
            'WetlandLoss', 'InadequatePlanning', 'PoliticalFactors'
        ]

        input_features = [float(request.form.get(f)) for f in feature_order]
        input_data = np.array(input_features).reshape(1, -1)

        if flood_model:
            prediction = flood_model.predict(input_data)[0]
            result = "🌊 Flood Likely" if prediction == 1 else "✅ No Flood Risk"
        else:
            result = "❌ Flood model not loaded"

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"result": f"❌ Error: {str(e)}"})

@app.route('/heatwave', methods=['GET', 'POST'])
def heatwave():
    prediction = None
    if request.method == 'POST':
        prediction = predict_heatwave(request.form)
    return render_template('pages/heatwave.html', prediction=prediction)

def predict_heatwave(form_data):
    try:
        max_temp = float(form_data['temperature_max'])
        min_temp = float(form_data['temperature_min'])
        max_humidity = float(form_data['humidity_max'])
        min_humidity = float(form_data['humidity_min'])
        precipitation = float(form_data['precipitation'])
        uv_index = float(form_data['uv_index'])
        visibility = float(form_data['visibility'])
        historical = 1.0 if form_data['historicalData'].lower() == 'yes' else 0.0

        urban_heat_map = {'low': 0.0, 'medium': 1.0, 'high': 2.0}
        urban_heat_input = form_data.get('urbanHeat', 'medium').lower()
        urban_heat_effect = urban_heat_map.get(urban_heat_input, 1.0)

        features = np.array([[max_temp, min_temp, max_humidity, min_humidity,
                              precipitation, uv_index, visibility,
                              historical, urban_heat_effect]])

        if heatwave_model:
            result = heatwave_model.predict(features)
            return "🔥 Heatwave Likely" if result[0] == 1 else "✅ No Heatwave"
        else:
            return "❌ Heatwave model not loaded"
    except Exception as e:
        return f"❌ Error: {str(e)}"

@app.route('/thunderstorm')
def thunderstorm():
    return render_template('pages/thunderstorm.html')

@app.route('/predict-thunderstorm', methods=['POST'])
def predict_thunderstorm():
    try:
        data = request.get_json()
        print("✅ Received Thunderstorm Data:", data)
        # 🔧 List of all input features expected from the frontend form
        
        THUNDERSTORM_FEATURES = [
            'temperature', 'dew_point', 'humidity', 'pressure',
            'wind_speed', 'wind_direction', 'cloud_cover', 'precipitation',
            'uvi', 'visibility', 'elevation', 'weather_code'
        ]


        # Dynamically fetch features using predefined list
        features = [float(data[feature]) for feature in THUNDERSTORM_FEATURES]
        input_array = np.array([features])
        print("📊 Thunderstorm Input Array:", input_array)

        prediction = thunderstorm_model.predict(input_array)[0]
        print("⚡ Thunderstorm Prediction Result:", prediction)

        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        print("❌ Error in thunderstorm prediction:", str(e))
        return jsonify({'error': str(e)}), 500







@app.route('/updates')
def updates():
    return render_template('pages/updates.html')

@app.route('/login')
def login():
    return render_template('pages/login.html')

@app.route('/signup')
def signup():
    return render_template('pages/signup.html')

# ---------------------- MAIN ----------------------
if __name__ == '__main__':
    app.run(debug=True)
