import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("dataset/dataset.csv")

# Encode target labels
label_encoder = LabelEncoder()
df["Target"] = label_encoder.fit_transform(df["Target"])

# Features and Target
X = df.drop("Target", axis=1)
y = df["Target"]

# --- Feature engineering -------------------------------------------------
# Raw "approved" counts let the model treat a light course load (e.g. 2
# enrolled / 2 approved) as weak in absolute terms, even though the pass
# RATE is perfect. Adding explicit pass-rate features gives the model
# direct access to that ratio instead of only the raw magnitude, so a
# strong-but-light course load isn't penalized as heavily.
X["Pass rate 1st sem"] = (
    X["Curricular units 1st sem (approved)"]
    / X["Curricular units 1st sem (enrolled)"].replace(0, pd.NA)
).fillna(0.0)

X["Pass rate 2nd sem"] = (
    X["Curricular units 2nd sem (approved)"]
    / X["Curricular units 2nd sem (enrolled)"].replace(0, pd.NA)
).fillna(0.0)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        eval_metric="mlogloss",
        random_state=42
    )
}

best_model = None
best_accuracy = 0

for name, model in models.items():

    print("\n" + "=" * 50)
    print(name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print(classification_report(y_test, predictions))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Save best model
joblib.dump(best_model, "saved_models/best_model.pkl")
joblib.dump(scaler, "saved_models/scaler.pkl")
joblib.dump(label_encoder, "saved_models/label_encoder.pkl")

print("\nBest Model Accuracy:", round(best_accuracy * 100, 2), "%")
print("Best model saved successfully!")