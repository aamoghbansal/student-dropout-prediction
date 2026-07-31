# 🎓 Smart Campus AI

An AI-powered web application that predicts whether a student is likely to **Graduate**, **Remain Enrolled**, or **Drop Out** using Machine Learning.

## 📌 Project Overview

Student dropout is a major challenge for educational institutions. This project uses the **UCI Student Dropout and Academic Success Dataset** to build a predictive model that helps identify students at risk based on their academic, personal, and socioeconomic information.

The application provides:
- Student status prediction
- Prediction confidence score
- Basic recommendation based on the prediction

---

## 🚀 Features

- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Multiple Machine Learning Models
- Random Forest Model Selection
- CNN Model Training
- Flask Web Application
- Responsive Bootstrap UI
- Real-time Student Status Prediction

---

## 🛠️ Technologies Used

- Python
- Flask
- HTML5
- Bootstrap 5
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- XGBoost
- Matplotlib
- Seaborn
- Joblib

---

## 📊 Dataset

**Dataset:** UCI Student Dropout and Academic Success Dataset

- 4,424 Student Records
- 34 Input Features
- Target Classes:
  - Graduate
  - Enrolled
  - Dropout

---

## 🤖 Machine Learning Models

We trained and compared the following models:

- Logistic Regression
- Decision Tree
- Random Forest ✅
- XGBoost
- Convolutional Neural Network (CNN)

### Best Model

**Random Forest**

**Accuracy:** ~77.97%

---

## 📂 Project Structure

```
SmartCampus-AI/
│
├── app.py
├── backend/
│   ├── 01_eda.py
│   ├── 02_preprocessing.py
│   ├── 03_train_ml.py
│   ├── 04_model_evaluation.py
│   └── 05_train_cnn.py
│
├── dataset/
│   └── dataset.csv
│
├── saved_models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── cnn_model.keras
│
├── graphs/
│   ├── confusion_matrix.png
│   ├── cnn_accuracy.png
│   └── cnn_loss.png
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/SmartCampus-AI.git
```

Move to the project folder:

```bash
cd SmartCampus-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open your browser:

```
http://127.0.0.1:5000
```

---

## 📈 Workflow

1. Load Dataset
2. Data Preprocessing
3. Exploratory Data Analysis
4. Train Multiple ML Models
5. Compare Model Performance
6. Select Best Model
7. Deploy using Flask
8. Predict Student Status

---

## 📷 Output

The application predicts whether the student is:

- 🎓 Graduate
- 📚 Enrolled
- ⚠️ Dropout

Along with:

- Confidence Score
- Recommendation

---

## 🔮 Future Improvements

- Interactive Dashboard
- Database Integration
- Student Login System
- Cloud Deployment
- Email Notification System
- Mobile Application

---

## 👨‍💻 Team

Final Year B.Tech CSE (AI & ML) Project

UPES, Dehradun

---

## 📄 License

This project is developed for educational purposes.
