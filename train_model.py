"""Train the fitness activity classifier and save it to model.pkl.

The dataset is synthetic (generated from typical sensor ranges per activity),
so the project runs end to end without any external data. Replace
`generate_data()` with your own CSV loader to train on real sensor data.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

FEATURES = ["heart_rate", "accel_mean", "accel_std", "cadence"]
MODEL_PATH = "model.pkl"
RANDOM_STATE = 42

# (mean, std) per feature for each activity
PROFILES = {
    "resting": {"heart_rate": (65, 6), "accel_mean": (1.00, 0.02), "accel_std": (0.02, 0.01), "cadence": (0, 2)},
    "walking": {"heart_rate": (95, 8), "accel_mean": (1.10, 0.06), "accel_std": (0.30, 0.07), "cadence": (105, 8)},
    "running": {"heart_rate": (155, 12), "accel_mean": (1.80, 0.20), "accel_std": (0.90, 0.15), "cadence": (170, 10)},
    "cycling": {"heart_rate": (130, 10), "accel_mean": (1.20, 0.08), "accel_std": (0.45, 0.10), "cadence": (80, 9)},
}


def generate_data(samples_per_class: int = 500, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label, profile in PROFILES.items():
        data = {f: rng.normal(*profile[f], samples_per_class) for f in FEATURES}
        df = pd.DataFrame(data)
        df[["accel_mean", "accel_std", "cadence"]] = df[["accel_mean", "accel_std", "cadence"]].clip(lower=0)
        df["activity"] = label
        rows.append(df)
    return pd.concat(rows, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def train(model_path: str = MODEL_PATH) -> float:
    df = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["activity"], test_size=0.2, stratify=df["activity"], random_state=RANDOM_STATE
    )
    model = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"Accuracy: {accuracy:.3f}")
    print(classification_report(y_test, preds))

    joblib.dump({"model": model, "features": FEATURES}, model_path)
    print(f"Model saved to {model_path}")
    return accuracy


if __name__ == "__main__":
    train()
