import sys
import os
import threading
import webbrowser
import time
from flask import Flask, render_template, request
import pickle
import numpy as np

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

app = Flask(__name__,
            template_folder=resource_path('templates'),
            static_folder=resource_path('static'))

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

            features_array = np.array(features).reshape(1, -1)
            scaled_features = scaler.transform(features_array)
            prediction = model.predict(scaled_features)

            if prediction[0] == 1:
                result = "High Risk of Diabetes"
                color = "red"
            else:
                result = "Low Risk of Diabetes"
                color = "green"

            return render_template('index.html', prediction_text=result, color=color)
        except Exception as e:
            return render_template('index.html', prediction_text=f"Error: {str(e)}", color="orange")

def open_browser():
    time.sleep(2)
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(debug=False, port=5000, use_reloader=False)