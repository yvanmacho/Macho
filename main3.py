from flask import Flask, request, render_template, jsonify
import joblib
import os
import numpy as np

app = Flask(__name__)

# Correct path to the model file
model_path = r'C:\Users\educa\Documents\Project T4S\api\random_forest_model.joblib'

# Load the model
if os.path.exists(model_path):
    model = joblib.load(open(model_path, "rb"))
else:
    print(f"Model file not found at path: {model_path}")
    model = None  # Handle case where model file is not found

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text='Model not loaded')
    try:
        # Retrieve form data
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        wind_speed = float(request.form['wind_speed'])
        weather_situation = request.form['weather_situation']
        month = int(request.form['month'])
        holiday = int(request.form['holiday'])
        week_day = int(request.form['week_day'])
        working_day = int(request.form['working_day'])
        season = request.form['season']

        # Create one-hot encoding for weather_situation and season
        weather_clear = 1 if weather_situation == 'clear' else 0
        weather_few_clouds = 1 if weather_situation == 'few clouds' else 0
        weather_partly_cloudy = 1 if weather_situation == 'partly cloudy' else 0
        season_fall = 1 if season == 'fall' else 0
        season_spring = 1 if season == 'spring' else 0
        season_summer = 1 if season == 'summer' else 0
        season_winter = 1 if season == 'winter' else 0

        # Create feature array
        features = np.array([[temperature, humidity, wind_speed, weather_clear, weather_few_clouds, weather_partly_cloudy,
                              month, holiday, week_day, working_day, season_fall, season_spring, season_summer, season_winter]])

        # Make prediction
        prediction = model.predict(features)

        # Render template with prediction
        return render_template('index.html', prediction_text=f'The predicted number of rentals is {prediction[0]}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
