import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None

from field_labels import (
    APPLICATION_MODE_OPTIONS,
    COURSE_OPTIONS,
    FRIENDLY_NAMES,
    MARITAL_OPTIONS,
    NATIONALITY_OPTIONS,
    PARENT_OCCUPATION_OPTIONS,
    PARENT_QUALIFICATION_OPTIONS,
    PREVIOUS_QUALIFICATION_OPTIONS,
)

app = Flask(__name__)

model = joblib.load("saved_models/best_model.pkl")
scaler = joblib.load("saved_models/scaler.pkl")
label_encoder = joblib.load("saved_models/label_encoder.pkl")
lstm_model = None
lstm_scaler = None

if load_model is not None:
    try:
        lstm_model = load_model("saved_models/lstm_model.keras")
        lstm_scaler = joblib.load("saved_models/lstm_scaler.pkl")
    except (OSError, FileNotFoundError):
        lstm_model = None
        lstm_scaler = None

PER_PAGE = 50
RANDOM_STATE = 42
METRICS_CACHE = None

# How much weight an OPTIONAL/additional-detail field's actual value carries,
# relative to its dataset-median default, when it differs from that default.
# 1.0 = full impact (original behavior). 0.0 = optional fields are fully
# ignored and always treated as the median. Lower this to relax how much a
# single unusual optional field can swing the prediction.
OPTIONAL_FIELD_WEIGHT = 0.35

FEATURE_ORDER = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Age at enrollment",
    "International",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

# Top predictors + enrolled counts so "subjects passed" is meaningful
REQUIRED_FIELDS = {
    "Course",
    "Gender",
    "Age at enrollment",
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
}

_df = pd.read_csv("dataset/dataset.csv")
FEATURE_DEFAULTS = {col: float(_df[col].median()) for col in FEATURE_ORDER}
del _df

FORM_OPTIONS = {
    "courses": COURSE_OPTIONS,
    "marital": MARITAL_OPTIONS,
    "application_modes": APPLICATION_MODE_OPTIONS,
    "previous_qualifications": PREVIOUS_QUALIFICATION_OPTIONS,
    "nationalities": NATIONALITY_OPTIONS,
    "parent_qualifications": PARENT_QUALIFICATION_OPTIONS,
    "parent_occupations": PARENT_OCCUPATION_OPTIONS,
}


def template_extras(**kwargs):
    payload = {
        "form_data": {},
        "defaults_used": 0,
        "drivers": [],
        **FORM_OPTIONS,
    }
    payload.update(kwargs)
    return payload


GRADE_FIELDS = {
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)",
}

# If left blank, copy from the matching enrolled count instead of dataset medians
EVAL_FROM_ENROLLED = {
    "Curricular units 1st sem (evaluations)": "Curricular units 1st sem (enrolled)",
    "Curricular units 2nd sem (evaluations)": "Curricular units 2nd sem (enrolled)",
}


def parse_optional_float(raw, default):
    if raw is None:
        return default, True
    text = str(raw).strip()
    if text == "":
        return default, True
    return float(text), False


def to_model_grade(sgpa):
    """Convert college SGPA (0–10) to the dataset grade scale (0–20)."""
    if sgpa < 0 or sgpa > 10:
        raise ValueError("SGPA must be between 0 and 10")
    return sgpa * 2.0


def build_feature_vector(form):
    # Resolve required values first so we can derive smarter optional defaults
    required_values = {}
    for name in REQUIRED_FIELDS:
        text = "" if form.get(name) is None else str(form.get(name)).strip()
        if text == "":
            raise ValueError(f"Missing required field: {FRIENDLY_NAMES.get(name, name)}")
        value = float(text)
        if name in GRADE_FIELDS:
            value = to_model_grade(value)
        required_values[name] = value

    for sem in ("1st", "2nd"):
        enrolled = required_values[f"Curricular units {sem} sem (enrolled)"]
        approved = required_values[f"Curricular units {sem} sem (approved)"]
        if approved > enrolled:
            raise ValueError(
                f"Subjects passed ({sem} semester) cannot be more than subjects enrolled"
            )

    values = []
    used_defaults = []
    for name in FEATURE_ORDER:
        if name in REQUIRED_FIELDS:
            values.append(required_values[name])
            continue

        raw = form.get(name)
        if name in EVAL_FROM_ENROLLED and (raw is None or str(raw).strip() == ""):
            value = required_values[EVAL_FROM_ENROLLED[name]]
            used_defaults.append(name)
        else:
            # Prefer 0 for sparse academic counters instead of unrelated medians
            fallback = FEATURE_DEFAULTS[name]
            if name.endswith("(credited)") or name.endswith("(without evaluations)"):
                fallback = 0.0
            value, was_default = parse_optional_float(raw, fallback)
            if was_default:
                used_defaults.append(name)
            else:
                # Relax the influence of this optional/additional field so a
                # single unusual value doesn't swing the prediction as hard
                # as a required academic/financial field would.
                value = fallback + OPTIONAL_FIELD_WEIGHT * (value - fallback)
        values.append(value)

    return np.array(values, dtype=float).reshape(1, -1), used_defaults


# The Random Forest was retrained with two extra engineered features (pass
# rate per semester) so it can see the RATE of subjects passed, not just the
# raw count. This stops a light-but-strong course load (e.g. 2 enrolled / 2
# approved) from being penalized the same way a heavy, weak one would be.
RF_FEATURE_ORDER = FEATURE_ORDER + ["Pass rate 1st sem", "Pass rate 2nd sem"]


def add_engineered_features(raw_features):
    frame = pd.DataFrame(raw_features, columns=FEATURE_ORDER)

    enrolled_1 = frame["Curricular units 1st sem (enrolled)"]
    approved_1 = frame["Curricular units 1st sem (approved)"]
    frame["Pass rate 1st sem"] = np.where(enrolled_1 > 0, approved_1 / enrolled_1, 0.0)

    enrolled_2 = frame["Curricular units 2nd sem (enrolled)"]
    approved_2 = frame["Curricular units 2nd sem (approved)"]
    frame["Pass rate 2nd sem"] = np.where(enrolled_2 > 0, approved_2 / enrolled_2, 0.0)

    return frame


def scale_features(raw_features):
    frame = add_engineered_features(raw_features)
    return scaler.transform(frame[RF_FEATURE_ORDER])


def scale_lstm_features(raw_features):
    frame = pd.DataFrame(raw_features, columns=FEATURE_ORDER)
    scaled = lstm_scaler.transform(frame)
    return scaled.reshape(scaled.shape[0], scaled.shape[1], 1)


def model_probabilities(raw_features, scaled_features):

    rf_probs = model.predict_proba(scaled_features)
    if lstm_model is None or lstm_scaler is None:
        return rf_probs, "Random Forest"

    # Always blend both models for the live prediction instead of picking
    # whichever one happened to score best on the overall test set. Silently
    # using only one model's raw output would discard the other model's
    # opinion on this specific case, and would make the "Final Prediction"
    # card just duplicate whichever single model won the global comparison.
    lstm_features = scale_lstm_features(raw_features)
    lstm_probs = lstm_model.predict(lstm_features, verbose=0)
    return (rf_probs + lstm_probs) / 2.0, "Random Forest + LSTM"

def all_model_predictions(raw_features, scaled_features):

    # Random Forest
    rf_probs = model.predict_proba(scaled_features)[0]

    rf_prediction = label_encoder.inverse_transform(
        [np.argmax(rf_probs)]
    )[0]

    rf_confidence = round(
        float(np.max(rf_probs))*100,
        2
    )


    # LSTM
    lstm_prediction = "Unavailable"
    lstm_confidence = 0
    lstm_probs_dict = {}

    if lstm_model is not None and lstm_scaler is not None:

        lstm_features = scale_lstm_features(
            raw_features
        )

        lstm_probs = lstm_model.predict(
            lstm_features,
            verbose=0
        )[0]

        lstm_prediction = label_encoder.inverse_transform(
            [np.argmax(lstm_probs)]
        )[0]

        lstm_confidence = round(
            float(np.max(lstm_probs))*100,
            2
        )

        lstm_probs_dict = {

            str(label): round(float(prob)*100,2)

            for label,prob in zip(
                label_encoder.classes_,
                lstm_probs
            )
        }


    # Ensemble

    ensemble_probs = (
        rf_probs + lstm_probs
    ) / 2

    ensemble_prediction = label_encoder.inverse_transform(
        [np.argmax(ensemble_probs)]
    )[0]

    ensemble_confidence = round(
        float(np.max(ensemble_probs))*100,
        2
    )


    return {

        "rf_prediction": rf_prediction,
        "rf_confidence": rf_confidence,

        "lstm_prediction": lstm_prediction,
        "lstm_confidence": lstm_confidence,

        "ensemble_prediction": ensemble_prediction,
        "ensemble_confidence": ensemble_confidence,

    }

def class_probabilities(scaled_features):
    probs = model.predict_proba(scaled_features)[0]
    return {
        str(label): round(float(prob) * 100, 1)
        for label, prob in zip(label_encoder.classes_, probs)
    }


def format_probabilities(probs):
    return {
        str(label): round(float(prob) * 100, 1)
        for label, prob in zip(label_encoder.classes_, probs)
    }


def student_progress(raw_features):
    values = raw_features[0]
    lookup = dict(zip(FEATURE_ORDER, values))

    enrolled_1 = lookup["Curricular units 1st sem (enrolled)"]
    enrolled_2 = lookup["Curricular units 2nd sem (enrolled)"]
    approved_1 = lookup["Curricular units 1st sem (approved)"]
    approved_2 = lookup["Curricular units 2nd sem (approved)"]

    enrolled_total = enrolled_1 + enrolled_2
    approved_total = approved_1 + approved_2
    pass_rate = approved_total / enrolled_total if enrolled_total else 0

    return {
        "pass_rate": pass_rate,
        "avg_sgpa": (
            lookup["Curricular units 1st sem (grade)"]
            + lookup["Curricular units 2nd sem (grade)"]
        ) / 4.0,
        "debtor": lookup["Debtor"],
        "tuition_current": lookup["Tuition fees up to date"],
    }


LIGHT_LOAD_THRESHOLD = 0.6  # flag if enrolled count is below 60% of the dataset median


def light_course_load_note(raw_features):
    values = raw_features[0]
    lookup = dict(zip(FEATURE_ORDER, values))

    enrolled_1 = lookup["Curricular units 1st sem (enrolled)"]
    enrolled_2 = lookup["Curricular units 2nd sem (enrolled)"]
    median_1 = FEATURE_DEFAULTS["Curricular units 1st sem (enrolled)"]
    median_2 = FEATURE_DEFAULTS["Curricular units 2nd sem (enrolled)"]

    is_light = (
        enrolled_1 < median_1 * LIGHT_LOAD_THRESHOLD
        or enrolled_2 < median_2 * LIGHT_LOAD_THRESHOLD
    )
    if not is_light:
        return ""

    return (
        f"Note: this student enrolled in fewer subjects per semester "
        f"({int(enrolled_1)} and {int(enrolled_2)}) than the typical student "
        f"({int(median_1)}-{int(median_2)}). A light course load, even with a strong "
        "pass rate, is uncommon in the training data, so this prediction carries more "
        "uncertainty than usual."
    )


def calibrated_prediction(raw_features, scaled_features):
    probs, model_source = model_probabilities(raw_features, scaled_features)
    row_probs = probs[0]
    result = label_encoder.inverse_transform([int(np.argmax(row_probs))])[0]
    probabilities = format_probabilities(row_probs)
    confidence = round(max(probabilities.values()), 2)

    progress = student_progress(raw_features)
    financially_clear = progress["debtor"] == 0 and progress["tuition_current"] == 1
    strong_academics = progress["pass_rate"] >= 0.70 and progress["avg_sgpa"] >= 6.0
    borderline_dropout = result == "Dropout" and probabilities.get("Dropout", 0) < 70

    review_note = ""
    if borderline_dropout and financially_clear and strong_academics:
        non_dropout = {
            label: prob
            for label, prob in probabilities.items()
            if label != "Dropout"
        }
        result = max(non_dropout, key=non_dropout.get)
        confidence = non_dropout[result]
        review_note = (
            "The model initially leaned toward Dropout, but the student's fees, debt "
            "status, SGPA, and pass rate indicate continued enrollment is more appropriate."
        )

    load_note = light_course_load_note(raw_features)
    if load_note:
        review_note = f"{review_note} {load_note}".strip() if review_note else load_note

    return result, confidence, probabilities, review_note, model_source


def model_status():
    if lstm_model is not None and lstm_scaler is not None:
        return "Random Forest + LSTM"
    return "Random Forest"


def model_status_note():
    if lstm_model is not None and lstm_scaler is not None:
        return "LSTM ensemble is active."
    if load_model is None:
        return "Install TensorFlow and train the LSTM model to activate the ensemble."
    return "Train the LSTM model to activate the ensemble."


def test_split():
    df = pd.read_csv("dataset/dataset.csv")
    x = df[FEATURE_ORDER]
    y = label_encoder.transform(df["Target"])
    return train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )[1::2]


def summarize_metrics(name, y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=range(len(label_encoder.classes_)),
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=range(len(label_encoder.classes_)),
    )

    return {
        "name": name,
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "macro_precision": round(macro[0] * 100, 2),
        "macro_recall": round(macro[1] * 100, 2),
        "macro_f1": round(macro[2] * 100, 2),
        "weighted_f1": round(weighted[2] * 100, 2),
        "classes": [
            {
                "name": str(label),
                "precision": round(float(precision[i]) * 100, 2),
                "recall": round(float(recall[i]) * 100, 2),
                "f1": round(float(f1[i]) * 100, 2),
                "support": int(support[i]),
            }
            for i, label in enumerate(label_encoder.classes_)
        ],
        "matrix": matrix.tolist(),
        "matrix_max": int(matrix.max()) if matrix.size else 1,
    }


def build_metrics_report():
    global METRICS_CACHE
    if METRICS_CACHE is not None:
        return METRICS_CACHE

    x_test, y_test = test_split()
    raw_features = x_test.to_numpy()
    scaled_features = scale_features(raw_features)

    rf_probs = model.predict_proba(scaled_features)
    reports = [
        summarize_metrics(
            "Random Forest",
            y_test,
            np.argmax(rf_probs, axis=1),
        )
    ]

    if lstm_model is not None and lstm_scaler is not None:
        lstm_features = scale_lstm_features(raw_features)
        lstm_probs = lstm_model.predict(lstm_features, verbose=0)
        reports.append(
            summarize_metrics(
                "LSTM",
                y_test,
                np.argmax(lstm_probs, axis=1),
            )
        )
        reports.append(
            summarize_metrics(
                "Random Forest + LSTM",
                y_test,
                np.argmax((rf_probs + lstm_probs) / 2.0, axis=1),
            )
        )

    best = max(reports, key=lambda item: item["macro_f1"])
    METRICS_CACHE = {
        "models": reports,
        "labels": [str(label) for label in label_encoder.classes_],
        "test_size": len(y_test),
        "best_model": best,
        "lstm_active": lstm_model is not None and lstm_scaler is not None,
    }
    return METRICS_CACHE


DRIVER_FRIENDLY_NAMES = {
    **FRIENDLY_NAMES,
    "Pass rate 1st sem": "Pass rate (1st semester)",
    "Pass rate 2nd sem": "Pass rate (2nd semester)",
}


def top_drivers(scaled_features, n=5):
    contrib = model.feature_importances_ * np.abs(scaled_features[0])
    ranked = sorted(
        zip(RF_FEATURE_ORDER, contrib),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {
            "name": DRIVER_FRIENDLY_NAMES.get(name, name),
            "score": round(float(score) * 100, 1),
        }
        for name, score in ranked[:n]
    ]


def load_student_predictions():
    df = pd.read_csv("dataset/dataset.csv")
    actual = df["Target"].to_numpy()

    features = scale_features(df[FEATURE_ORDER].to_numpy())
    preds = label_encoder.inverse_transform(model.predict(features))
    confidences = np.round(np.max(model.predict_proba(features), axis=1) * 100, 2)

    ages = df["Age at enrollment"].astype(int).to_numpy()
    genders = np.where(df["Gender"].to_numpy() == 1, "Male", "Female")
    scholarships = np.where(df["Scholarship holder"].to_numpy() == 1, "Yes", "No")
    debtors = np.where(df["Debtor"].to_numpy() == 1, "Yes", "No")

    students = []
    for i in range(len(df)):
        students.append({
            "id": i + 1,
            "age": int(ages[i]),
            "gender": genders[i],
            "scholarship": scholarships[i],
            "debtor": debtors[i],
            "actual": actual[i],
            "predicted": preds[i],
            "confidence": float(confidences[i]),
            "match": actual[i] == preds[i],
        })
    return students


def risk_summary(all_students):
    predicted_dropout = [s for s in all_students if s["predicted"] == "Dropout"]
    high_risk = [s for s in predicted_dropout if s["confidence"] >= 70]
    return {
        "predicted_graduate": sum(1 for s in all_students if s["predicted"] == "Graduate"),
        "predicted_enrolled": sum(1 for s in all_students if s["predicted"] == "Enrolled"),
        "predicted_dropout": len(predicted_dropout),
        "high_risk": len(high_risk),
        "mismatches": sum(1 for s in all_students if not s["match"]),
    }


@app.route("/")
def home():
    return render_template(
        "index.html",
        **template_extras(
            model_source=model_status(),
            model_note=model_status_note(),
        ),
    )


@app.route("/students")
def students():
    all_students = load_student_predictions()

    status_filter = request.args.get("status", "all")
    result_filter = request.args.get("result", "all")
    risk_filter = request.args.get("risk", "all")
    page = max(1, request.args.get("page", 1, type=int))

    filtered = all_students
    if status_filter != "all":
        filtered = [s for s in filtered if s["actual"] == status_filter]
    if result_filter == "correct":
        filtered = [s for s in filtered if s["match"]]
    elif result_filter == "incorrect":
        filtered = [s for s in filtered if not s["match"]]
    if risk_filter == "dropout":
        filtered = [s for s in filtered if s["predicted"] == "Dropout"]
    elif risk_filter == "high":
        filtered = [
            s for s in filtered
            if s["predicted"] == "Dropout" and s["confidence"] >= 70
        ]

    total = len(filtered)
    correct = sum(1 for s in all_students if s["match"])
    accuracy = round(correct / len(all_students) * 100, 2) if all_students else 0
    summary = risk_summary(all_students)

    total_pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)
    page = min(page, total_pages)
    start = (page - 1) * PER_PAGE
    page_students = filtered[start:start + PER_PAGE]

    return render_template(
        "students.html",
        students=page_students,
        page=page,
        total_pages=total_pages,
        total=total,
        dataset_size=len(all_students),
        accuracy=accuracy,
        status_filter=status_filter,
        result_filter=result_filter,
        risk_filter=risk_filter,
        summary=summary,
    )


@app.route("/metrics")
def metrics():
    return render_template("metrics.html", **build_metrics_report())


@app.route("/predict", methods=["POST"])
def predict():
    form_data = {key: request.form.get(key, "") for key in FEATURE_ORDER}

    try:
        raw_features, used_defaults = build_feature_vector(request.form)
        scaled = scale_features(raw_features)
        all_results = all_model_predictions(raw_features, scaled)

        result, confidence, probabilities, review_note, model_source = calibrated_prediction(
            raw_features,
            scaled,
        )
        drivers = top_drivers(scaled)

        return render_template(
            "index.html",
            **template_extras(
                prediction=result,
                confidence=confidence,
                probabilities=probabilities,
                review_note=review_note,
                model_source=model_source,
                model_note=model_status_note(),
                form_data=form_data,
                defaults_used=len(used_defaults),
                drivers=drivers,
                rf_prediction=all_results["rf_prediction"],
rf_confidence=all_results["rf_confidence"],

lstm_prediction=all_results["lstm_prediction"],
lstm_confidence=all_results["lstm_confidence"],

ensemble_prediction=all_results["ensemble_prediction"],
                ensemble_confidence=all_results["ensemble_confidence"],
                final_prediction=result,
                final_confidence=confidence,
            ),
        )

    except Exception as e:
        return render_template(
            "index.html",
            **template_extras(
                prediction="Prediction Failed",
                confidence="",
                probabilities={},
                review_note="",
                model_source=model_status(),
                model_note=model_status_note(),
                error=str(e),
                form_data=form_data,
                defaults_used=0,
                drivers=[],
            ),
        )


if __name__ == "__main__":
    app.run(debug=True)