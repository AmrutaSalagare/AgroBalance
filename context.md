# Project Context: Agricultural Assistant

## Overview

**Agricultural Assistant** is a smart, end-to-end precision farming solution that leverages real-time soil sensor data and machine learning models to recommend optimal crops and fertilizers for farmers. The system is designed to automate and enhance agricultural decision-making by integrating hardware (soil NPK sensor), data preprocessing, and advanced ML inference within a user-friendly web interface.

---

## Key Features

- **Real-time Soil Sensing:** Reads soil parameters (N, P, K, pH, moisture, temperature) directly from a Modbus RTU-compatible sensor.
- **ML-based Crop Recommendation:** Suggests the most suitable crop for current soil and environmental conditions using trained neural network and random forest models.
- **Fertilizer Recommendation:** Recommends the best fertilizer type and application rate, tailored to the predicted crop and soil status.
- **Dual Input Mode:** Supports both automatic sensor-based input and manual data entry for flexibility.
- **Web Interface:** Clean, responsive UI for data input, sensor reading, and displaying recommendations.
- **Sensor Testing Utility:** Standalone script to verify sensor connectivity and output.

---

## System Architecture

### 1. **Hardware Layer**

- **Soil Sensor:** Measures Nitrogen (N), Phosphorus (P), Potassium (K), pH, moisture, and temperature.
- **Connection:** Communicates via serial port (default: `COM3`) using the Modbus RTU protocol.

### 2. **Backend (Flask Application)**

- **Sensor Integration:** Reads sensor data using `minimalmodbus` and `pyserial`.
- **ML Inference:** Loads pre-trained models and encoders from the `models/` directory.
- **API Endpoints:**
  - `/read_sensor`: Returns live sensor readings as JSON.
  - `/predict`: Accepts form data (from sensor or manual input), runs ML models, and returns recommendations.
- **Business Logic:** Maps crops to fertilizer categories, calculates fertilizer amounts based on soil efficiency factors, and handles all preprocessing.

### 3. **Frontend**

- **HTML/CSS/JS:** Located in `templates/index.html` and `static/styles.css`.
- **Sensor Controls:** "Read from Sensor" button fetches and auto-fills form fields.
- **Form Submission:** User can submit either sensor-populated or manually entered data for recommendations.
- **Results Display:** Shows recommended crop, fertilizer type, and suggested application rate.

---

## Machine Learning Models

- **Crop Recommendation:**
  - `crop_model.keras` (Neural Network)
  - `rf_crop_model.pkl` (Random Forest)
  - `scaler_crop.pkl`, `encoder_crop.pkl`, `crop_type_encoder.pkl` (Preprocessing)
- **Fertilizer Recommendation:**
  - `fertilizer_model.keras` (Neural Network)
  - `rf_fertilizer_model.pkl` (Random Forest)
  - `scaler_fertilizer.pkl`, `fertilizer_label_encoder.pkl`, `soil_type_encoder.pkl` (Preprocessing)
- **Training Data:** `crop_recommendation.csv`, `fertilizer_recommendation.csv`
- **Training Notebook:** `Training.ipynb` contains all model training and export code.

---

## Data Flow

1. **Sensor Data Acquisition:**
   - User clicks "Read from Sensor" → `/read_sensor` endpoint → sensor values returned and auto-filled in the form.
2. **Manual Data Entry:**
   - User enters soil/environmental data manually in the form fields.
3. **Prediction:**
   - On form submission, `/predict` endpoint processes the data.
   - Data is preprocessed (scaling, encoding).
   - Crop is predicted using ML model.
   - Fertilizer is predicted using ML model, considering crop, soil, and environmental factors.
   - Fertilizer amount is calculated based on crop requirements, soil efficiency, and fertilizer composition.
4. **Results:**
   - Recommendations are rendered on the main page.

---

## File Structure

```
├── app.py                      # Main Flask application
├── check.py                    # Sensor testing utility
├── requirements.txt            # Python dependencies
├── crop_recommendation.csv     # Training data for crop models
├── fertilizer_recommendation.csv # Training data for fertilizer models
├── Training.ipynb              # Jupyter notebook with model training code
├── models/                     # Trained ML models and preprocessing objects
│   ├── crop_model.keras
│   ├── fertilizer_model.keras
│   ├── rf_crop_model.pkl
│   ├── rf_fertilizer_model.pkl
│   ├── scaler_crop.pkl
│   ├── scaler_fertilizer.pkl
│   ├── encoder_crop.pkl
│   ├── fertilizer_label_encoder.pkl
│   ├── soil_type_encoder.pkl
│   └── crop_type_encoder.pkl
├── static/
│   └── styles.css              # Custom CSS for UI
└── templates/
    └── index.html              # Main HTML template
```

---

## Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Test sensor connection:**
   ```bash
   python check.py
   ```
3. **Run the web application:**
   ```bash
   python app.py
   ```
4. **Access the UI:**  
   Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Sensor Testing (`check.py`)

- Use `check.py` to verify sensor output and troubleshoot hardware issues.
- Prints all raw and interpreted sensor values to the console.

---

## Customization

- **Sensor Port:** Change the serial port in `app.py` and `check.py` if your sensor is not on `COM3`.
- **Model Retraining:** Use `Training.ipynb` to retrain models with new data and export updated model files.
- **UI:** Modify `templates/index.html` and `static/styles.css` for branding or layout changes.

---

## Dependencies

- Flask
- TensorFlow / Keras
- scikit-learn
- minimalmodbus
- pyserial
- numpy
- joblib

---

## Limitations & Notes

- Sensor must be connected and powered for real-time readings.
- Humidity and rainfall are not measured by the sensor and must be entered manually.
- The system assumes the sensor register mapping as per the current implementation.
- Model accuracy depends on the quality and representativeness of the training data.

---

## Authors & License

- Developed by [Your Name/Team]
- License: [Your License Here]

---

## Contact

For support or contributions, please contact [your-email@example.com].
