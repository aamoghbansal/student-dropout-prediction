import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


DATASET_PATH = "dataset/dataset.csv"
MODEL_PATH = "saved_models/lstm_model.keras"
SCALER_PATH = "saved_models/lstm_scaler.pkl"
ENCODER_PATH = "saved_models/lstm_label_encoder.pkl"
ACCURACY_GRAPH_PATH = "graphs/lstm_accuracy.png"
LOSS_GRAPH_PATH = "graphs/lstm_loss.png"
CONFUSION_MATRIX_PATH = "graphs/lstm_confusion_matrix.png"
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATASET_PATH)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Target"])
    x = df.drop("Target", axis=1)
    return x, y, label_encoder


def build_model(input_shape, class_count):
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.25),
            LSTM(32),
            Dropout(0.25),
            Dense(32, activation="relu"),
            Dense(class_count, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("LSTM Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ACCURACY_GRAPH_PATH)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_GRAPH_PATH)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("LSTM Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=35, ha="right")
    plt.yticks(tick_marks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            plt.text(
                col,
                row,
                cm[row, col],
                ha="center",
                va="center",
                color="white" if cm[row, col] > cm.max() / 2 else "black",
            )

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()


def main():
    x, y, label_encoder = load_data()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    model = build_model(
        input_shape=(x_train.shape[1], x_train.shape[2]),
        class_count=y_train_cat.shape[1],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=0.00001,
        ),
    ]

    history = model.fit(
        x_train,
        y_train_cat,
        validation_split=0.20,
        epochs=80,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    loss, accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    print("LSTM Accuracy:", round(accuracy * 100, 2), "%")
    print("LSTM Accuracy Check:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    plot_history(history)
    plot_confusion_matrix(y_test, y_pred, label_encoder.classes_)

    print("LSTM model saved successfully.")
    print("Saved:", MODEL_PATH)
    print("Saved:", SCALER_PATH)
    print("Saved:", ENCODER_PATH)


if __name__ == "__main__":
    main()
