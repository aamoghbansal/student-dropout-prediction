# Smart Campus AI — Student Dropout Prediction

A machine-learning powered Flask application that predicts whether a student is likely to **Graduate, remain Enrolled, or Drop Out** based on academic, demographic, socioeconomic, and related enrollment information.

The project combines classical machine learning and deep learning models with a web-based prediction interface, model evaluation dashboard, dataset-wide prediction browser, and prediction explanations.

## Overview

The system was developed to explore how machine learning can be used to identify student outcomes and potentially highlight students who may require additional academic or financial support.

The application works with the **UCI Student Dropout and Academic Success Dataset**, containing 4,424 student records and 34 input features.

## Key Features

- Student outcome prediction
- Three target classes:
  - Graduate
  - Enrolled
  - Dropout
- Academic and demographic input form
- Probability for each prediction class
- Feature-importance based explanation
- Borderline dropout calibration logic
- Dataset-wide prediction browser
- Filtering by actual status, correctness, and risk
- Pagination
- Model metrics dashboard
- Accuracy, precision, recall, and F1 evaluation
- Confusion matrices
- Comparison of classical ML models
- CNN and LSTM deep-learning experiments
- Optional Random Forest + LSTM ensemble
- Responsive Bootstrap interface

## Machine Learning Models

The project trains and compares:

| Model | Role |
|---|---|
| Logistic Regression | Baseline classifier |
| Decision Tree | Tree-based comparison |
| Random Forest | Primary production model |
| XGBoost | Gradient boosting comparison |
| CNN | Deep learning experiment |
| LSTM | Deep learning / optional ensemble |

The live application can select the available model configuration based on held-out test performance using macro F1.

## Tech Stack

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Joblib
- Matplotlib
- Seaborn
- HTML5
- Bootstrap 5

## ML Workflow

```text
Dataset
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Preprocessing
   │
   ├── Encoding
   ├── Scaling
   └── Train/Test Split
   │
   ▼
Model Training
   │
   ├── Logistic Regression
   ├── Decision Tree
   ├── Random Forest
   ├── XGBoost
   ├── CNN
   └── LSTM
   │
   ▼
Model Evaluation
   │
   ▼
Best Model / Ensemble
   │
   ▼
Flask Prediction Application
```

## Project Structure

```text
student-dropout-prediction/
├── app.py
├── field_labels.py
├── requirements.txt
├── backend/
│   ├── 01_eda.py
│   ├── 02_preprocessing.py
│   ├── 03_train_ml.py
│   ├── 04_model_evaluation.py
│   ├── 05_train_cnn.py
│   └── 06_train_lstm.py
├── dataset/
│   └── dataset.csv
├── saved_models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── cnn_model.keras
│   └── lstm_model.keras
├── graphs/
├── templates/
│   ├── index.html
│   ├── students.html
│   └── metrics.html
└── static/
    └── style.css
```

## Dataset

The project uses the UCI Student Dropout and Academic Success Dataset:

- **4,424 records**
- **34 input features**
- Target classes:
  - Graduate: 2,209
  - Dropout: 1,421
  - Enrolled: 794

Features include academic performance, enrollment information, demographic characteristics, financial indicators, and other contextual variables.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/aamoghbansal/student-dropout-prediction.git
cd student-dropout-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Main Routes

| Route | Purpose |
|---|---|
| `/` | Student prediction form |
| `/predict` | Prediction submission endpoint |
| `/students` | Dataset-wide prediction browser |
| `/metrics` | Model evaluation dashboard |

## Explainability

For individual predictions, the application can display the predicted outcome, class probabilities, and important features contributing to the prediction.

This makes the system more useful than a simple black-box classifier because users can inspect factors associated with the prediction.

## Future Improvements

- Persistent student database
- Authentication and role-based access
- Cloud deployment
- Automated alerts for high-risk students
- Improved model explainability using SHAP
- Model versioning
- Automated retraining pipeline
- CNN integration into the live ensemble
- Monitoring for model drift

## Project Context

Final Year B.Tech CSE project — UPES, Dehradun.

## License

Developed for educational purposes.

## Author

**Aamogh Bansal**

GitHub: https://github.com/aamoghbansal
