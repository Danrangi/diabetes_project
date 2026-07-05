# 🏥 Diabetes Risk Prediction System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Framework-Flask-brightgreen.svg)]()
[![ML Model](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)]()

> An intelligent machine learning system that predicts diabetes risk based on patient medical data. Early detection, better health outcomes.

---

## 📋 Overview

**Diabetes Risk Prediction System** is a web-based diagnostic tool that uses machine learning to assess the risk of diabetes based on key health metrics. The system analyzes patient medical data and provides instant predictions to help with early screening and medical decision-making.

This application is powered by a trained machine learning model that evaluates:
- **Pregnancies** — Number of pregnancies
- **Glucose** — Fasting blood glucose level
- **Blood Pressure** — Systolic blood pressure
- **Skin Thickness** — Triceps skin fold thickness
- **Insulin** — 2-Hour serum insulin
- **BMI** — Body Mass Index
- **Diabetes Pedigree Function** — Family history genetic risk
- **Age** — Patient age in years

---

## 🎯 Key Features

- ✅ **Instant Predictions** — Get diabetes risk assessment in seconds
- 🔬 **Medical-Grade Model** — Trained on real clinical data
- 💻 **User-Friendly Interface** — Simple, intuitive web application
- 📊 **Accurate Results** — High-precision predictions with visual feedback
- 🔒 **Local Processing** — All data processed locally on your device
- 🚀 **Fast & Lightweight** — Minimal dependencies, quick startup

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Web browser (Chrome, Firefox, Safari, Edge)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Danrangi/diabetes_project.git
   cd diabetes_project
   ```

2. **Install required dependencies**
   ```bash
   pip install flask numpy scikit-learn
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the application**
   - The web browser will automatically open
   - If not, navigate to: `http://localhost:5000`

---

## 📁 Project Structure

```
diabetes_project/
├── README.md                    # Project documentation
├── app.py                       # Flask application & predictions
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── diabetes_model.pkl           # Trained ML model
├── scaler.pkl                   # Feature scaler (normalization)
├── templates/
│   └── index.html               # Web interface
├── static/
│   ├── css/                     # Styling
│   └── js/                      # Frontend scripts
└── .github/
    └── workflows/               # CI/CD configurations
```

---

## 💡 How It Works

The prediction system follows a simple 4-step process:

```
[Patient Input] → [Data Validation] → [Feature Scaling] → [ML Model] → [Risk Assessment]
```

### Step-by-Step Process:

1. **Input Collection** — User enters 8 key health metrics
2. **Validation** — System validates all input values
3. **Normalization** — Features are scaled using pre-trained scaler
4. **Prediction** — Machine learning model processes normalized features
5. **Result** — System returns risk level (High Risk 🔴 or Low Risk 🟢)

### What the Model Does:
- Takes 8 health parameters as input
- Applies feature scaling to normalize values
- Runs prediction through trained classifier
- Returns binary classification (Diabetic Risk: Yes/No)

---

## 📊 Input Parameters Explained

| Parameter | Range | Description |
|-----------|-------|-------------|
| **Pregnancies** | 0+ | Number of times pregnant (women) |
| **Glucose** | 70-200+ | Fasting blood glucose in mg/dL |
| **Blood Pressure** | 60-180 | Systolic blood pressure in mmHg |
| **Skin Thickness** | 0-100 | Triceps skin fold in mm |
| **Insulin** | 0-900+ | 2-hour serum insulin in mIU/ml |
| **BMI** | 18-60+ | Body Mass Index (weight/height²) |
| **Diabetes Pedigree** | 0.0-3.0 | Genetic diabetes predisposition |
| **Age** | 18-120 | Age in years |

---

## 🛠️ Technologies Used

- **Framework**: Flask (Python web framework)
- **Machine Learning**: scikit-learn
- **Data Processing**: NumPy, pandas
- **Frontend**: HTML5, CSS3, JavaScript
- **Model Type**: Supervised Learning Classifier
- **Serialization**: pickle

---

## 📈 Model Information

- **Training Dataset**: Pima Indians Diabetes Database
- **Samples**: 768 patient records
- **Features**: 8 medical parameters
- **Output Classes**: 2 (Diabetic Risk: Yes/No)
- **Algorithm**: Logistic Regression / Support Vector Machine
- **Preprocessing**: StandardScaler normalization

---

## 🖥️ User Interface

The web application provides:

- **Clean Input Form** — Easy-to-use form with labeled fields
- **Real-time Validation** — Immediate feedback on input errors
- **Visual Results** — Color-coded risk assessment
  - 🔴 **Red** — High Risk of Diabetes
  - 🟢 **Green** — Low Risk of Diabetes
- **Responsive Design** — Works on desktop, tablet, and mobile

---

## 💻 Usage Example

### Via Web Interface:
1. Open `http://localhost:5000`
2. Enter patient health metrics
3. Click "Predict" button
4. View instant risk assessment

### Via Python Script:
```python
import pickle
import numpy as np

# Load model and scaler
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Patient data: [Pregnancies, Glucose, BloodPressure, SkinThickness, 
#                Insulin, BMI, DiabetesPedigreeFunction, Age]
patient_data = np.array([[1, 89, 66, 23, 94, 28.1, 0.167, 21]])

# Scale and predict
scaled_data = scaler.transform(patient_data)
prediction = model.predict(scaled_data)

print("High Risk" if prediction[0] == 1 else "Low Risk")
```

---

## ⚙️ Configuration

### Port Configuration
To run on a different port, edit `app.py`:
```python
app.run(debug=False, port=8000)  # Change 5000 to desired port
```

### Model Loading
The application automatically:
- Detects the execution environment
- Loads models from correct paths (works with PyInstaller)
- Handles lazy loading for optimal performance

---

## 📝 Dependencies

See `requirements.txt`:
```
Flask==2.3.0
numpy==1.24.0
scikit-learn==1.3.0
Werkzeug==2.3.0
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## ⚠️ Disclaimer

This tool is designed for **informational and educational purposes only**. It should **NOT** be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical advice, diagnosis, and treatment.

---

## 🤝 Contributing

Contributions are welcome! To improve this project:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/improvement`)
3. **Commit** your changes (`git commit -m 'Add improvement'`)
4. **Push** to the branch (`git push origin feature/improvement`)
5. **Open** a Pull Request

### Areas for Contribution:
- Improved accuracy with better models
- Enhanced UI/UX design
- Mobile application version
- Additional health metrics
- Data visualization features
- Documentation improvements

---

## 📊 Model Performance Metrics

Expected performance on validation data:
- **Accuracy**: 75-80%
- **Precision**: High for positive class
- **Recall**: Balanced across classes
- **Specificity**: Good true negative rate

*Note: Actual performance may vary based on dataset*

---

## 🔒 Data Privacy

- ✅ No data is stored or transmitted
- ✅ All predictions happen locally
- ✅ No internet connection required after startup
- ✅ No tracking or analytics

---

## 📞 Support & Contact

Have questions or issues?

- **GitHub Issues**: [Report a bug](https://github.com/Danrangi/diabetes_project/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/Danrangi/diabetes_project/discussions)
- **Email**: Contact via GitHub profile

---

## 📚 References & Resources

- [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Medical Literature on Diabetes](https://www.who.int/health-topics/diabetes)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Pima Indians Diabetes Database (UCI Machine Learning Repository)
- scikit-learn and Flask communities
- All contributors and users providing feedback

---

## 📈 Roadmap

Future enhancements planned:
- [ ] Mobile app version (iOS/Android)
- [ ] Advanced visualization dashboard
- [ ] Multiple ML model comparison
- [ ] Patient history tracking
- [ ] Real-time model updates
- [ ] API endpoint for integration
- [ ] Multi-language support
- [ ] Dark mode UI

---

**Made with ❤️ for better health awareness and early diabetes detection**

*Last Updated: 2026*
