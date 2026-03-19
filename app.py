import sys
import os
from flask import Flask, render_template, request
import pickle
import numpy as np

# FUNCTION: Handle paths for PyInstaller (The secret to offline EXE)
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Initialize Flask with explicit folder paths so it finds HTML and CSS
app = Flask(__name__, 
            template_folder=resource_path('templates'), 
            static_folder=resource_path('static'))

# Load the trained model and scaler using the resource_path function
model_path = resource_path('diabetes_model.pkl')
scaler_path = resource_path('scaler.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Retrieve form data and convert to float
            features = [
                float(request.form['Pregnancies']),
                float(request.form['Glucose']),
                float(request.form['BloodPressure']),
                float(request.form['SkinThickness']),
                float(request.form['Insulin']),
                float(request.form['BMI']),
                float(request.form['DiabetesPedigreeFunction']),
                float(request.form['Age'])
            ]
            
            # Convert to 2D numpy array
            features_array = np.array(features).reshape(1, -1)
            
            # Scale the data using the saved scaler
            scaled_features = scaler.transform(features_array)
            
            # Make prediction
            prediction = model.predict(scaled_features)
            
            # Determine the result message and color
            if prediction[0] == 1:
                result = "High Risk of Diabetes"
                color = "red"
            else:
                result = "Low Risk of Diabetes"
                color = "green"
                
            return render_template('index.html', prediction_text=result, color=color)
        except Exception as e:
            return render_template('index.html', prediction_text=f"Error: {str(e)}", color="orange")

if __name__ == '__main__':
    # Run the app locally on port 5000
    app.run(debug=False, port=5000)
