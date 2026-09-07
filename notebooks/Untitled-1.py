# %%
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy

MOVEMENT_DIR = Path(
    r"C:\Users\Sheel\OneDrive\Documents\Default Project\data\pads-parkinsons-disease-smartwatch-dataset-1.0.0\pads-parkinsons-disease-smartwatch-dataset-1.0.0\preprocessed\movement"
)

FS = 100.0

file_list = pd.read_csv(MOVEMENT_DIR.parent / "file_list.csv")

tasks = [
    "Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2",
    "StretchHold", "HoldWeight", "DrinkGlas", "CrossArms",
    "TouchNose", "Entrainment1", "Entrainment2",
]
wrists = ["LeftWrist", "RightWrist"]
sensors = ["Accelerometer", "Gyroscope"]
axes = ["X", "Y", "Z"]
channels = [f"{t}_{w}_{s}_{a}" for t in tasks for w in wrists for s in sensors for a in axes]

print(len(channels), "channels")

# %%
bin_files = sorted(MOVEMENT_DIR.glob("*_ml.bin"))
ids = np.array([int(f.name.split("_")[0]) for f in bin_files])
movement = np.stack([np.fromfile(f, dtype=np.float32).reshape(len(channels), -1) for f in bin_files])
print("movement:", movement.shape, movement.dtype)
print("ids aligned with file_list:", np.array_equal(ids, file_list["id"].values))

# %%
groups = {
    "Healthy": [52, 193, 1, 46, 455, 254, 389, 204, 45, 108, 449],
    "Multiple Sclerosis": [17, 36, 44, 98, 142, 195, 225, 301, 421, 432, 438],
    "Parkinson's": [38, 307, 286, 299, 175, 115, 4, 119, 252, 237, 71],
    "Atypical Parkinsonism": [397, 64, 238, 304, 114, 291, 110, 168, 32, 15, 18],
    "Essential Tremor": [100, 200, 278, 72, 143, 104, 112, 126, 423, 124, 66],
}
id_to_group = {pid: g for g, pids in groups.items() for pid in pids}

mask = np.isin(ids, list(id_to_group))
mov = movement[mask]
pids = ids[mask]
print("participants:", len(pids))
print(pd.Series([id_to_group[p] for p in pids]).value_counts().to_string())

# %%
channel_df = pd.DataFrame({
    "channel": channels,
    "task": [c.split("_")[0] for c in channels],
    "wrist": [c.split("_")[1] for c in channels],
    "sensor": [c.split("_")[2] for c in channels],
    "axis": [c.split("_")[3] for c in channels],
})

predictor_names = [
    "Primary Frequency Component",
    "Acceleration Norm",
    "Jerk",
    "Signal Change Variability",
    "Spectral Entropy",
    "Mean Velocity",
    "Relative Tremor Power",
    "Harmonic Power Ratio",
    "Movement Amplitude",
]

stream_feats = []
feature_names = []

eps = 1e-12
for (task, wrist), grp in channel_df.groupby(["task", "wrist"]):
    acc = mov[:, grp.index[grp["sensor"] == "Accelerometer"].tolist(), :]
    acc_norm = np.linalg.norm(acc, axis=1)

    n = acc_norm.shape[1]
    freqs = np.fft.rfftfreq(n, d=1 / FS)
    power = np.abs(np.fft.rfft(acc_norm, axis=1)) ** 2
    jerk = np.diff(acc_norm, axis=1)

    # Velocity proxy from IMU acceleration magnitude.
    velocity = np.cumsum(acc_norm - acc_norm.mean(axis=1, keepdims=True), axis=1) / FS
    mean_velocity = np.mean(np.abs(velocity), axis=1)

    tremor_band = (freqs >= 4.0) & (freqs <= 8.0)
    total_power = power[:, 1:].sum(axis=1) + eps
    tremor_power = power[:, tremor_band].sum(axis=1)
    relative_tremor_power = tremor_power / total_power

    if tremor_band.any():
        tremor_power_matrix = power[:, tremor_band]
        main_idx = np.argmax(tremor_power_matrix, axis=1)
        main_freqs = freqs[tremor_band][main_idx]
        main_power = tremor_power_matrix[np.arange(power.shape[0]), main_idx] + eps

        harmonic_power = np.zeros(power.shape[0], dtype=np.float64)
        for harmonic in (2.0, 3.0):
            target_freq = main_freqs * harmonic
            nearest_idx = np.abs(freqs[None, :] - target_freq[:, None]).argmin(axis=1)
            harmonic_power += power[np.arange(power.shape[0]), nearest_idx]
        harmonic_power_ratio = harmonic_power / main_power
    else:
        harmonic_power_ratio = np.zeros(power.shape[0], dtype=np.float64)

    values = [
        freqs[1:][np.argmax(power[:, 1:], axis=1)],
        acc_norm.mean(axis=1),
        jerk.mean(axis=1),
        jerk.std(axis=1),
        entropy(power[:, 1:] + eps, axis=1),
        mean_velocity,
        relative_tremor_power,
        harmonic_power_ratio,
        np.ptp(acc_norm, axis=1),
    ]

    for name, vec in zip(predictor_names, values):
        stream_feats.append(vec)
        feature_names.append(f"{task}_{wrist}_{name}")

X = np.column_stack(stream_feats)
y = np.array([id_to_group[p] for p in pids])
print("X:", X.shape, "| y:", y.shape)
print(pd.Series(y).value_counts().to_string())

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

binary = np.isin(y, ["Healthy", "Parkinson's"])
Xb, yb = X[binary], y[binary]
yb = (yb == "Parkinson's").astype(int)

Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    Xb, yb, test_size=0.3, stratify=yb, random_state=0
)

model = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(solver="saga", l1_ratio=0.5, max_iter=5000, random_state=0)),
])
model.fit(Xb_train, yb_train)

print("PD vs HC accuracy:", accuracy_score(yb_test, model.predict(Xb_test)))
print(classification_report(yb_test, model.predict(Xb_test), target_names=["HC", "Parkinson's"]))

# %%
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

group_names = list(groups)
y_codes = pd.Categorical(y, categories=group_names).codes

# 1. Input your data
X = X  # feature matrix (participants x predictors)
y = y_codes  # outcome as integer class codes (0..4)

# 2. Set up cross-validation for a small dataset
# Stratified 3-Fold ensures all 5 classes are represented in every split
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
# 3. Create a pipeline to scale features and fit the model
# Elastic net requires feature scaling to penalize coefficients fairly
# (sklearn 1.9: scale and multi_class were removed; multinomial is automatic
#  and the penalty type is implied by l1_ratio)
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegressionCV(
        cv=cv,
        solver='saga',
        l1_ratios=[0.1, 0.5, 0.7, 0.9],
        Cs=10,
        scoring='accuracy',
        use_legacy_attributes=True,
        random_state=42,
        max_iter=5000,
    ))
])

# 4. Fit the model
model_pipeline.fit(X, y)

# 5. Extract results
best_clf = model_pipeline.named_steps['clf']
print(f"Optimal L1 Ratio (Alpha): {best_clf.l1_ratio_}")
print(f"Optimal C (1/Lambda): {best_clf.C_}")
print(f"Mean CV Accuracy Score: {best_clf.scores_[1].mean():.2f}")

# %%
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split


group_names = list(groups)
y_labels = np.array([id_to_group[p] for p in pids])
label_codes = pd.Categorical(y_labels, categories=group_names).codes

# 1. Input your data
X_model = X  # feature matrix (participants x predictors)
y_model = label_codes  # outcome as integer class codes (0..4)

# 2. Set up cross-validation for a small dataset
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# 3. Create a pipeline to scale features and fit the model
model_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegressionCV(
        cv=cv,
        penalty="elasticnet",
        solver="saga",
        l1_ratios=[0.1, 0.5, 0.7, 0.9],
        Cs=10,
        scoring="accuracy",
        use_legacy_attributes=True,
        random_state=42,
        max_iter=5000,
    ))
])

# 4. Fit the model
model_pipeline.fit(X_model, y_model)

# 5. Extract CV results
best_clf = model_pipeline.named_steps["clf"]
mean_cv_accuracy = np.mean([scores.mean() for scores in best_clf.scores_.values()])

print(f"Optimal L1 Ratio (Alpha): {best_clf.l1_ratio_}")
print(f"Optimal C (1/Lambda): {best_clf.C_}")
print(f"Mean CV Accuracy Score: {mean_cv_accuracy:.2f}")

# Additional observed CV accuracy
observed_scores = cross_val_score(model_pipeline, X_model, y_model, cv=cv, scoring="accuracy")
observed_accuracy = observed_scores.mean()
print(f"Observed CV Accuracy: {observed_accuracy:.2f}")

# Holdout report for precision/recall/F1/support
X_train, X_test, y_train, y_test = train_test_split(
    X_model,
    y_model,
    test_size=0.3,
    stratify=y_model,
    random_state=42,
)
model_pipeline.fit(X_train, y_train)
predictions = model_pipeline.predict(X_test)
print(f"5-class accuracy: {accuracy_score(y_test, predictions):.2f}")
print(classification_report(y_test, predictions, target_names=group_names))

print("X shape:", X_model.shape)
print("Number of participants:", len(y_model))
print("\nClass counts:")
print(np.unique(y_model, return_counts=True))


