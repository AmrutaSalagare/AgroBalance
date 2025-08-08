from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import minimalmodbus
import serial
import time
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load models and preprocessing objects
crop_model = load_model('./models/crop_model.keras')
fertilizer_model = load_model('./models/fertilizer_model.keras')
rf_crop_model = joblib.load('./models/rf_crop_model.pkl')
rf_fertilizer_model = joblib.load('./models/rf_fertilizer_model.pkl')
scaler_crop = joblib.load('./models/scaler_crop.pkl')
scaler_fertilizer = joblib.load('./models/scaler_fertilizer.pkl')
encoder_crop = joblib.load('./models/encoder_crop.pkl')
fertilizer_label_encoder = joblib.load('./models/fertilizer_label_encoder.pkl')
soil_type_encoder = joblib.load('./models/soil_type_encoder.pkl')
crop_type_encoder = joblib.load('./models/crop_type_encoder.pkl')

# Crop mapping for fertilizer recommendation
crop_mapping = {
    'rice': 'rice',
    'maize': 'Maize',
    'chickpea': 'Pulses',
    'kidneybeans': 'kidneybeans',
    'pigeonpeas': 'Pulses',
    'mothbeans': 'Pulses',
    'mungbean': 'Pulses',
    'blackgram': 'Pulses',
    'lentil': 'Pulses',
    'pomegranate': 'pomegranate',
    'banana': 'pomegranate',
    'mango': 'pomegranate',
    'grapes': 'pomegranate',
    'watermelon': 'watermelon',
    'muskmelon': 'watermelon',
    'apple': 'pomegranate',
    'orange': 'orange',
    'papaya': 'pomegranate',
    'coconut': 'Oil seeds',
    'cotton': 'Cotton',
    'jute': 'Cotton',
    'coffee': 'coffee'
}

# NPK recommendations and fertilizer compositions
crop_original_recommendations = {
    'rice': {'N': '100-120 kg/ha', 'P': '40-50 kg/ha', 'K': '40-60 kg/ha'},
    'maize': {'N': '120-150 kg/ha', 'P': '50-60 kg/ha', 'K': '30-40 kg/ha'},
    'chickpea': {'N': '20-30 kg/ha', 'P': '40-50 kg/ha', 'K': '20-30 kg/ha'},
    'kidneybeans': {'N': '25-30 kg/ha', 'P': '50-60 kg/ha', 'K': '30-40 kg/ha'},
    'pigeonpeas': {'N': '20-25 kg/ha', 'P': '40-50 kg/ha', 'K': '20-30 kg/ha'},
    'mothbeans': {'N': '15-20 kg/ha', 'P': '30-40 kg/ha', 'K': '15-20 kg/ha'},
    'mungbean': {'N': '20-25 kg/ha', 'P': '40-50 kg/ha', 'K': '20-30 kg/ha'},
    'blackgram': {'N': '15-20 kg/ha', 'P': '30-40 kg/ha', 'K': '15-20 kg/ha'},
    'lentil': {'N': '25-30 kg/ha', 'P': '40-50 kg/ha', 'K': '20-30 kg/ha'},
    'pomegranate': {'N': '70-80 kg/ha', 'P': '30-40 kg/ha', 'K': '50-60 kg/ha'},
    'banana': {'N': '150-180 kg/ha', 'P': '50-60 kg/ha', 'K': '150-200 kg/ha'},
    'mango': {'N': '90-100 kg/ha', 'P': '30-40 kg/ha', 'K': '80-100 kg/ha'},
    'grapes': {'N': '100-120 kg/ha', 'P': '30-40 kg/ha', 'K': '140-160 kg/ha'},
    'watermelon': {'N': '70-80 kg/ha', 'P': '30-40 kg/ha', 'K': '70-80 kg/ha'},
    'muskmelon': {'N': '70-80 kg/ha', 'P': '30-40 kg/ha', 'K': '70-80 kg/ha'},
    'apple': {'N': '70-80 kg/ha', 'P': '30-40 kg/ha', 'K': '50-60 kg/ha'},
    'orange': {'N': '80-90 kg/ha', 'P': '30-40 kg/ha', 'K': '60-70 kg/ha'},
    'papaya': {'N': '90-100 kg/ha', 'P': '30-40 kg/ha', 'K': '90-100 kg/ha'},
    'coconut': {'N': '90-100 kg/ha', 'P': '40-50 kg/ha', 'K': '100-120 kg/ha'},
    'cotton': {'N': '70-80 kg/ha', 'P': '30-40 kg/ha', 'K': '30-40 kg/ha'},
    'jute': {'N': '40-50 kg/ha', 'P': '20-30 kg/ha', 'K': '20-30 kg/ha'},
    'coffee': {'N': '100-120 kg/ha', 'P': '30-40 kg/ha', 'K': '70-80 kg/ha'}
}

fertilizer_npk_composition = {
    'Urea': {'N': 46, 'P': 0, 'K': 0},
    'TSP': {'N': 0, 'P': 46, 'K': 0},
    'Superphosphate': {'N': 0, 'P': 20, 'K': 0},
    'Potassium sulfate': {'N': 0, 'P': 0, 'K': 50},
    'Potassium chloride': {'N': 0, 'P': 0, 'K': 60},
    'DAP': {'N': 18, 'P': 46, 'K': 0},
    '28-28': {'N': 28, 'P': 28, 'K': 0},
    '20-20': {'N': 20, 'P': 20, 'K': 0},
    '17-17-17': {'N': 17, 'P': 17, 'K': 17},
    '15-15-15': {'N': 15, 'P': 15, 'K': 15},
    '14-35-14': {'N': 14, 'P': 35, 'K': 14},
    '14-14-14': {'N': 14, 'P': 14, 'K': 14},
    '10-26-26': {'N': 10, 'P': 26, 'K': 26},
    '10-10-10': {'N': 10, 'P': 10, 'K': 10}
}
# Efficiency factors for different soil types
soil_efficiency_factors = {
    'Sandy': {'N': 0.7, 'P': 0.5, 'K': 0.6},
    'Clayey': {'N': 1.2, 'P': 1.1, 'K': 1.3},
    'Red': {'N': 0.8, 'P': 0.7, 'K': 0.8},
    'Black': {'N': 1.0, 'P': 1.0, 'K': 1.1},
    'Loamy': {'N': 1.0, 'P': 1.0, 'K': 1.0}
}


def calculate_fertilizer(crop, soil_levels, fertilizer, soil_type):
    if crop not in crop_original_recommendations:
        raise ValueError(f"No NPK recommendations available for crop: {crop}")
    original_npk = crop_original_recommendations[crop]

    def parse_recommendation(value):
        value = value.replace("kg/ha", "").strip()
        if '-' in value:
            low, high = map(int, value.split('-'))
            return (low + high) / 2
        return int(value)

    recommendation_N = parse_recommendation(original_npk['N'])
    recommendation_P = parse_recommendation(original_npk['P'])
    recommendation_K = parse_recommendation(original_npk['K'])

    if fertilizer not in fertilizer_npk_composition:
        raise ValueError(
            f"No NPK composition available for fertilizer: {fertilizer}")
    fertilizer_npk = fertilizer_npk_composition[fertilizer]

    if soil_type not in soil_efficiency_factors:
        raise ValueError(
            f"No efficiency factors available for soil type: {soil_type}")
    efficiency = soil_efficiency_factors[soil_type]

    # Adjust recommendations based on soil efficiency
    adjusted_N = recommendation_N / efficiency['N']
    adjusted_P = recommendation_P / efficiency['P']
    adjusted_K = recommendation_K / efficiency['K']

    deficit_N = max(adjusted_N - soil_levels['N'], 0)
    deficit_P = max(adjusted_P - soil_levels['P'], 0)
    deficit_K = max(adjusted_K - soil_levels['K'], 0)

    fert_amount_N = deficit_N / \
        (fertilizer_npk['N'] / 100) if fertilizer_npk['N'] > 0 else 0
    fert_amount_P = deficit_P / \
        (fertilizer_npk['P'] / 100) if fertilizer_npk['P'] > 0 else 0
    fert_amount_K = deficit_K / \
        (fertilizer_npk['K'] / 100) if fertilizer_npk['K'] > 0 else 0

    return max(fert_amount_N, fert_amount_P, fert_amount_K)


def read_soil_sensor():
    """Read data from soil sensor and return as dictionary"""
    try:
        instrument = minimalmodbus.Instrument('COM3', 1)
        instrument.serial.baudrate = 9600
        instrument.serial.bytesize = 8
        instrument.serial.parity = serial.PARITY_NONE
        instrument.serial.stopbits = 1
        instrument.serial.timeout = 2
        instrument.mode = minimalmodbus.MODE_RTU

        time.sleep(2)  # Give sensor time to stabilize

        data = instrument.read_registers(0, 14)

        return {
            'Temperature': data[0] / 10.0,     # Temperature in °C
            'Moisture': data[2] / 10.0,        # Moisture in %
            'pH': data[4] / 10.0,              # pH value
            'N': data[1] / 10.0,               # N in mg/kg
            'P': data[5] / 10.0,               # P in mg/kg
            'K': data[6] / 10.0                # K in mg/kg
        }
    except Exception as e:
        return {'error': str(e)}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/read_sensor', methods=['GET'])
def get_sensor_data():
    sensor_data = read_soil_sensor()
    if 'error' in sensor_data:
        return jsonify({'success': False, 'error': sensor_data['error']})
    return jsonify({'success': True, 'data': sensor_data})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    try:
        # Extract raw features from the form
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['Temperature'])
        humidity = float(data['Humidity'])
        ph = float(data['pH'])
        rainfall = float(data['Rainfall'])
        moisture = float(data['Moisture'])
        soil_type = data['Soil Type']

        # Create raw input array for crop prediction
        crop_input_raw = np.array(
            [[N, P, K, temperature, humidity, ph, rainfall]])

        # Scale the input for the crop model
        scaled_crop_input = scaler_crop.transform(crop_input_raw)

        # Use RF model to predict crop
        rf_crop_pred_label = rf_crop_model.predict(scaled_crop_input)[0]

        # Convert integer label to crop name string
        predicted_crop = encoder_crop.inverse_transform(
            [rf_crop_pred_label])[0]
        crop_lower = predicted_crop.lower()

        # Map to fertilizer crop categories
        mapped_crop = crop_mapping.get(crop_lower, crop_lower)

        # IMPORTANT: Match the exact 6 features that the fertilizer scaler expects
        # This is likely the correct order based on common scenarios, but check your training code
        fertilizer_input_raw = np.array([[
            N, P, K,
            temperature, humidity, moisture
            # Note: not including soil_type and crop_type in scaling
        ]])

        # Scale only the 6 numeric features
        scaled_numeric_features = scaler_fertilizer.transform(
            fertilizer_input_raw)

        # For the final prediction, we'll use the RF model directly with the features in correct format
        # This assumes the RF model was trained with encoded categorical features after the scaled numeric ones
        soil_type_encoded = soil_type_encoder.transform([soil_type])[0]
        crop_type_encoded = crop_type_encoder.transform([mapped_crop])[0]

        # Model could expect different ordering - adjust based on your training code
        rf_fertilizer_input = np.column_stack((
            scaled_numeric_features,
            np.array([[soil_type_encoded, crop_type_encoded]])
        ))

        fertilizer_label = rf_fertilizer_model.predict(rf_fertilizer_input)[0]
        predicted_fertilizer = fertilizer_label_encoder.inverse_transform(
            [fertilizer_label])[0]

        # For demonstration, create a sample recommendation
        fertilizer_amount = f"Based on your soil and crop conditions: Apply {predicted_fertilizer} at recommended rates."

        return render_template('index.html', results={
            'crop': predicted_crop,
            'fertilizer': predicted_fertilizer,
            'amount': fertilizer_amount
        })

    except ValueError as e:
        return render_template('index.html', error=f"Invalid input value: {str(e)}")
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return render_template('index.html', error=f"Prediction error: {str(e)}\n{error_details}")


if __name__ == '__main__':
    app.run(debug=True)
