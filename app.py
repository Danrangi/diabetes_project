from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
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

if __name__ == '__main__':
    app.run(debug=True)
