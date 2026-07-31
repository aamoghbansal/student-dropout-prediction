# 🎓 Smart Campus AI

A Flask web app that predicts whether a student is likely to **Graduate**, remain **Enrolled**, or **Drop Out**, using a Random Forest model (with an optional LSTM ensemble) trained on academic, demographic, and socioeconomic data.

## 📌 Overview

Smart Campus AI takes a student's enrollment details and first/second semester academic performance and estimates their likely outcome. Beyond a single prediction form, it includes a dataset-wide predictions browser and a live model metrics dashboard.

Core capabilities:
- Predict a student's outcome from a form of academic + demographic inputs
- Show per-class probabilities and the top features driving the prediction
- Flag borderline "Dropout" calls that a student's financial/academic standing may not support, and explain the override
- Browse predictions for the entire dataset, with filters for actual status, correctness, and risk level
- View live-computed accuracy/precision/recall/F1 metrics per model on a held-out test split

## 🚀 Features

- Data preprocessing & exploratory data analysis
- Multiple ML models trained and compared (Logistic Regression, Decision Tree, Random Forest, XGBoost)
- Deep learning models: CNN and LSTM
- Random Forest as the primary production model, with an optional Random Forest + LSTM ensemble (auto-selected based on whichever scores highest on macro F1)
- Calibration logic that reviews borderline dropout predictions against fee/debt status and academic performance
- Feature-importance based explanation ("top drivers") for each prediction
- SGPA input on a 0–10 scale, converted internally to the dataset's 0–20 grade scale
- Dataset-wide predictions table with pagination and filtering (`/students`)
- Live model metrics dashboard with confusion matrices per model (`/metrics`)
- Responsive Bootstrap UI

## 🛠️ Technologies Used

- Python, Flask
- Pandas, NumPy
- Scikit-learn, XGBoost
- TensorFlow / Keras (CNN, LSTM)
- Matplotlib, Seaborn
- Joblib
- HTML5, Bootstrap 5

## 📊 Dataset

**Dataset:** UCI Student Dropout and Academic Success Dataset

- 4,424 student records
- 34 input features (marital status, application mode, course, prior qualification, parental background, financial status, per-semester curricular unit stats, macroeconomic indicators, etc.)
- Target classes: **Graduate** (2,209), **Dropout** (1,421), **Enrolled** (794)

## 🤖 Machine Learning Models

Trained and compared in `backend/`:

- Logistic Regression
- Decision Tree
- Random Forest ✅ (primary model, `saved_models/best_model.pkl`)
- XGBoost
- Convolutional Neural Network (`backend/05_train_cnn.py` — trained, saved to `saved_models/cnn_model.keras`, not currently wired into the live app)
- LSTM (`backend/06_train_lstm.py` — optionally ensembled with Random Forest at inference time if `saved_models/lstm_model.keras` is present)

At request time, the app builds a metrics report on a held-out test split for each available model and automatically serves predictions from whichever model (Random Forest, LSTM, or the RF+LSTM average) scores best on macro F1.

## 📂 Project Structure

```
student-dropout-prediction/
│
├── app.py                     # Flask app: routes, prediction, metrics, calibration logic
├── field_labels.py            # Dropdown options & friendly display names for form fields
├── requirements.txt
│
├── backend/
│   ├── 01_eda.py               # Exploratory data analysis
│   ├── 02_preprocessing.py     # Encoding, train/test split
│   ├── 03_train_ml.py          # Logistic Regression, Decision Tree, Random Forest, XGBoost
│   ├── 04_model_evaluation.py  # Confusion matrix / evaluation plots
│   ├── 05_train_cnn.py         # CNN training
│   └── 06_train_lstm.py        # LSTM training
│
├── dataset/
│   └── dataset.csv
│
├── saved_models/
│   ├── best_model.pkl          # Random Forest
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── cnn_model.keras
│   ├── lstm_model.keras
│   ├── lstm_scaler.pkl
│   └── lstm_label_encoder.pkl
│
├── graphs/
│   ├── confusion_matrix.png
│   ├── cnn_accuracy.png
│   ├── cnn_loss.png
│   ├── lstm_accuracy.png
│   ├── lstm_loss.png
│   └── lstm_confusion_matrix.png
│
├── templates/
│   ├── index.html              # Prediction form + result
│   ├── students.html           # Dataset-wide predictions browser
│   └── metrics.html            # Model metrics dashboard
│
├── static/
│   └── style.css
│
└── README.md
```

## ⚙️ Installation

```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```
python app.py
```

Open your browser at:

```
http://127.0.0.1:5000
```

## 🧭 Pages / Routes

| Route | Description |
|---|---|
| `/` | Prediction form — enter student details and get a predicted outcome with confidence and top drivers |
| `/predict` | Handles form submission (POST) |
| `/students` | Browse dataset-wide predictions with filters (status, correctness, risk) and pagination |
| `/metrics` | Live model metrics dashboard (accuracy, precision, recall, F1, confusion matrix) per model |

## 📈 Workflow

1. Load & explore dataset
2. Preprocess (encode target, scale features, train/test split)
3. Train and compare Logistic Regression, Decision Tree, Random Forest, XGBoost
4. Train CNN and LSTM deep learning models
5. Select best-performing model(s) — Random Forest, with optional LSTM ensemble
6. Serve predictions via Flask, with calibration checks and explainability

## 🔮 Future Improvements

- Wire the trained CNN into the live ensemble selection
- Database integration for persistent student records
- Student/staff login system
- Cloud deployment
- Email notification system for high-risk students
- Mobile application

## 👨‍💻 Team

Final Year B.Tech CSE Project — UPES, Dehradun

## 📄 License

This project is developed for educational purposes.
