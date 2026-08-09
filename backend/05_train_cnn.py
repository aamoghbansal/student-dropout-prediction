import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("dataset/dataset.csv")

# Encode Target
label_encoder = LabelEncoder()
df["Target"] = label_encoder.fit_transform(df["Target"])

# Features & Labels
X = df.drop("Target", axis=1).values
y = df["Target"].values

# Standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# CNN expects 3D input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print("Training Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)

# -----------------------------
# CNN Model
# -----------------------------
model = Sequential()

model.add(Conv1D(
    filters=32,
    kernel_size=3,
    activation="relu",
    input_shape=(34,1)
))

model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(64, activation="relu"))

model.add(Dropout(0.3))

model.add(Dense(3, activation="softmax"))

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Early Stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)

# -----------------------------
# Evaluate
# -----------------------------
loss, accuracy = model.evaluate(X_test, y_test)

print("\nCNN Accuracy:", round(accuracy*100,2), "%")

# Save Model
model.save("saved_models/cnn_model.keras")

print("\nCNN Model Saved Successfully!")

# -----------------------------
# Accuracy Graph
# -----------------------------
plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Accuracy")

plt.legend()

plt.savefig("graphs/cnn_accuracy.png")

plt.show()

# -----------------------------
# Loss Graph
# -----------------------------
plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Loss")

plt.legend()

plt.savefig("graphs/cnn_loss.png")

plt.show()