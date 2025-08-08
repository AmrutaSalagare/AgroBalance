# Agricultural Assistant

An intelligent system for precision farming that provides crop recommendations and fertilizer suggestions based on soil sensor data and environmental factors.

## Features

- **Real-time Soil Sensing**: Direct integration with soil NPK sensors
- **Intelligent Crop Recommendations**: ML-powered crop suggestions based on soil conditions
- **Precise Fertilizer Recommendations**: Smart fertilizer type and application rate suggestions
- **Dual Input Mode**: Use soil sensor readings or manual data entry

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/AmrutaSalagare/AgroBalance.git
   cd agricultural-assistant
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure your sensor is properly connected:
   - Default configuration is set to `COM3` port
   - Uses Modbus RTU protocol

## Usage

1. Start the Flask application:

   ```bash
   python app.py
   ```

2. Open your browser and navigate to:

   ```
   http://127.0.0.1:5000
   ```

3. Using the application:
   - Click "Read from Sensor" to automatically populate soil data fields
   - Or manually enter soil properties and environmental conditions
   - Click "Get Recommendations" to receive crop and fertilizer suggestions

## System Requirements

- Python 3.8+
- Serial port access for soil sensor communication
- Compatible soil NPK sensor with Modbus RTU support

## Technology Stack

- **Backend**: Flask, Python
- **Machine Learning**: TensorFlow, scikit-learn
- **Models**: Neural Networks and Random Forest classifiers
- **Sensor Communication**: MinimalModbus, PySerial
- **Frontend**: HTML, CSS, JavaScript

## Models

The system uses multiple machine learning models:

- Neural network model for crop recommendation
- Neural network model for fertilizer recommendation
- Random forest model for crop recommendation
- Random forest model for fertilizer recommendation

These models are trained on agricultural datasets considering soil properties, environmental conditions, and crop requirements.

## Testing

You can test the soil sensor connection independently using:

```bash
python check.py
```

This will read and display raw sensor readings to verify proper hardware communication.

## File Structure

```
├── app.py                      # Main Flask application
├── check.py                    # Sensor testing utility
├── requirements.txt            # Python dependencies
├── crop_recommendation.csv     # Training data for crop models
├── fertilizer_recommendation.csv # Training data for fertilizer models
├── Training.ipynb              # Jupyter notebook with model training code
├── models/                     # Trained ML models
│   ├── crop_model.keras
│   ├── fertilizer_model.keras
│   ├── rf_crop_model.pkl
│   └── rf_fertilizer_model.pkl
│   └── various encoder and scaler files
├── static/                     # Static assets
│   └── styles.css
└── templates/                  # HTML templates
    └── index.html
```



