import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("dataset/dataset.csv")

print("Original Shape:", df.shape)

# Encode target column
label_encoder = LabelEncoder()
df["Target"] = label_encoder.fit_transform(df["Target"])

print("\nClass Mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{label} --> {i}")

# Features and Target
X = df.drop("Target", axis=1)
y = df["Target"]

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

print("\nTraining Shape:", X_train.shape)
print("Testing Shape :", X_test.shape)