from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Assuming the input is a JSON object with keys: wind_speed, wind_direction, ambient_temperature, output_power
    features = [[
        data['wind_speed'],
        data['wind_direction'],
        data['ambient_temperature'],
        data['output_power']
    ]]
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
