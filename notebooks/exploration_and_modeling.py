# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.makedirs("sensor-activity-classifier/reports", exist_ok=True)

# %% Set the base path to your dataset
DATA_PATH = r"C:\Users\phant\Downloads\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset"

# %% Load one raw inertial signal file (body acceleration, X-axis, training set)
signal_path = DATA_PATH + r"\train\Inertial Signals\body_acc_x_train.txt"

body_acc_x = pd.read_csv(signal_path, sep=r"\s+", header=None)

print("Shape:", body_acc_x.shape)
print(body_acc_x.head())

# %% Load the activity labels (one label per window)
labels_path = DATA_PATH + r"\train\y_train.txt"
y_train = pd.read_csv(labels_path, header=None, names=["activity_id"])

print("Shape:", y_train.shape)
print(y_train.head())

# %% Load human-readable activity names and map them
activity_labels_path = DATA_PATH + r"\activity_labels.txt"
activity_names = pd.read_csv(activity_labels_path, sep=r"\s+", header=None,
                               names=["activity_id", "activity_name"])

print(activity_names)

# %% Map numeric labels to names
activity_map = dict(zip(activity_names.activity_id, activity_names.activity_name))
y_train["activity_name"] = y_train["activity_id"].map(activity_map)

print(y_train.head())
print(y_train.activity_name.value_counts())

# %% Find one window index for WALKING and one for SITTING
walking_idx = y_train[y_train.activity_name == "WALKING"].index[0]
sitting_idx = y_train[y_train.activity_name == "SITTING"].index[0]

print("Walking window index:", walking_idx)
print("Sitting window index:", sitting_idx)

# %% Plot both windows side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(body_acc_x.iloc[walking_idx])
axes[0].set_title("Body Acceleration X — WALKING")
axes[0].set_xlabel("Sample index (within window)")
axes[0].set_ylabel("Acceleration")

axes[1].plot(body_acc_x.iloc[sitting_idx])
axes[1].set_title("Body Acceleration X — SITTING")
axes[1].set_xlabel("Sample index (within window)")
axes[1].set_ylabel("Acceleration")

plt.tight_layout()
plt.show()

# %% Find window indices for harder-to-distinguish activity pairs
upstairs_idx = y_train[y_train.activity_name == "WALKING_UPSTAIRS"].index[0]
downstairs_idx = y_train[y_train.activity_name == "WALKING_DOWNSTAIRS"].index[0]
standing_idx = y_train[y_train.activity_name == "STANDING"].index[0]

print("Upstairs index:", upstairs_idx)
print("Downstairs index:", downstairs_idx)
print("Standing index:", standing_idx)

# %% Plot WALKING_UPSTAIRS vs WALKING_DOWNSTAIRS
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(body_acc_x.iloc[upstairs_idx])
axes[0].set_title("Body Acceleration X — WALKING UPSTAIRS")
axes[0].set_xlabel("Sample index")
axes[0].set_ylabel("Acceleration")

axes[1].plot(body_acc_x.iloc[downstairs_idx])
axes[1].set_title("Body Acceleration X — WALKING DOWNSTAIRS")
axes[1].set_xlabel("Sample index")
axes[1].set_ylabel("Acceleration")

plt.tight_layout()
plt.show()

# %% Plot STANDING vs SITTING (two "static" activities — the real test)
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(body_acc_x.iloc[standing_idx])
axes[0].set_title("Body Acceleration X — STANDING")
axes[0].set_xlabel("Sample index")
axes[0].set_ylabel("Acceleration")

axes[1].plot(body_acc_x.iloc[sitting_idx])
axes[1].set_title("Body Acceleration X — SITTING")
axes[1].set_xlabel("Sample index")
axes[1].set_ylabel("Acceleration")

plt.tight_layout()
plt.show()

# %% Plot multiple STANDING windows to see if patterns are consistent
standing_indices = y_train[y_train.activity_name == "STANDING"].index[:5]
sitting_indices = y_train[y_train.activity_name == "SITTING"].index[:5]

fig, axes = plt.subplots(2, 5, figsize=(20, 6))

for i, idx in enumerate(standing_indices):
    axes[0, i].plot(body_acc_x.iloc[idx])
    axes[0, i].set_title(f"STANDING #{i}")

for i, idx in enumerate(sitting_indices):
    axes[1, i].plot(body_acc_x.iloc[idx])
    axes[1, i].set_title(f"SITTING #{i}")

plt.tight_layout()
plt.show()

# %% Compare standard deviation of body_acc_x across standing vs sitting windows
standing_all_idx = y_train[y_train.activity_name == "STANDING"].index
sitting_all_idx = y_train[y_train.activity_name == "SITTING"].index

standing_std = body_acc_x.iloc[standing_all_idx].std(axis=1)
sitting_std = body_acc_x.iloc[sitting_all_idx].std(axis=1)

print("Standing std — mean:", standing_std.mean(), "| std of std:", standing_std.std())
print("Sitting std — mean:", sitting_std.mean(), "| std of std:", sitting_std.std())

# %% Plot distributions of std values for standing vs sitting
plt.figure(figsize=(8, 5))
plt.hist(standing_std, bins=30, alpha=0.5, label="STANDING")
plt.hist(sitting_std, bins=30, alpha=0.5, label="SITTING")
plt.xlabel("Standard deviation of window")
plt.ylabel("Number of windows")
plt.legend()
plt.title("Distribution of signal variance: Standing vs Sitting")
plt.show()

# %% Load gyroscope data and compare standing vs sitting variance
gyro_path = DATA_PATH + r"\train\Inertial Signals\body_gyro_x_train.txt"
body_gyro_x = pd.read_csv(gyro_path, sep=r"\s+", header=None)

standing_gyro_std = body_gyro_x.iloc[standing_all_idx].std(axis=1)
sitting_gyro_std = body_gyro_x.iloc[sitting_all_idx].std(axis=1)

print("Standing gyro std — mean:", standing_gyro_std.mean())
print("Sitting gyro std — mean:", sitting_gyro_std.mean())

plt.figure(figsize=(8, 5))
plt.hist(standing_gyro_std, bins=30, alpha=0.5, label="STANDING")
plt.hist(sitting_gyro_std, bins=30, alpha=0.5, label="SITTING")
plt.xlabel("Standard deviation of gyroscope signal")
plt.ylabel("Number of windows")
plt.legend()
plt.title("Gyroscope variance: Standing vs Sitting")
plt.show()

# %% Define a feature extraction function for one window (one row of signal)
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft

def extract_features(window):
    """
    window: a 1D array of 128 raw signal readings (one window, one axis)
    returns: a dictionary of computed features
    """
    features = {}

    # Time-domain features
    features['mean'] = np.mean(window)
    features['std'] = np.std(window)
    features['min'] = np.min(window)
    features['max'] = np.max(window)
    features['range'] = features['max'] - features['min']
    features['skew'] = skew(window)
    features['kurtosis'] = kurtosis(window)

    # Zero-crossing rate: how often the signal crosses its own mean
    mean_centered = window - features['mean']
    zero_crossings = np.sum(np.diff(np.sign(mean_centered)) != 0)
    features['zero_crossing_rate'] = zero_crossings

    # Signal magnitude area (sum of absolute values, normalized)
    features['sma'] = np.sum(np.abs(window)) / len(window)

    # Frequency-domain features (FFT)
    fft_vals = np.abs(fft(window))
    fft_vals = fft_vals[:len(fft_vals)//2]  # keep only positive frequencies
    features['dominant_freq_magnitude'] = np.max(fft_vals)
    features['spectral_energy'] = np.sum(fft_vals**2) / len(fft_vals)

    return features

# %% Sanity check: run it on one walking window and one sitting window
walking_features = extract_features(body_acc_x.iloc[walking_idx].values)
sitting_features = extract_features(body_acc_x.iloc[sitting_idx].values)

print("WALKING features:", walking_features)
print()
print("SITTING features:", sitting_features)

# %% Load all 6 raw signal axes for training set
signal_types = ['body_acc_x', 'body_acc_y', 'body_acc_z',
                'body_gyro_x', 'body_gyro_y', 'body_gyro_z']

raw_signals = {}

for sig in signal_types:
    path = DATA_PATH + rf"\train\Inertial Signals\{sig}_train.txt"
    raw_signals[sig] = pd.read_csv(path, sep=r"\s+", header=None)
    print(f"{sig}: {raw_signals[sig].shape}")

# %% Build the full feature table
num_windows = raw_signals['body_acc_x'].shape[0]
all_features = []

for i in range(num_windows):
    row_features = {}
    for sig in signal_types:
        window = raw_signals[sig].iloc[i].values
        feats = extract_features(window)
        # prefix each feature name with the signal type, e.g. "body_acc_x_mean"
        for key, value in feats.items():
            row_features[f"{sig}_{key}"] = value
    all_features.append(row_features)

    if i % 1000 == 0:
        print(f"Processed {i}/{num_windows} windows")

feature_df = pd.DataFrame(all_features)
print(feature_df.shape)
feature_df.head()

# %% Load subject IDs for training set
subject_path = DATA_PATH + r"\train\subject_train.txt"
subject_train = pd.read_csv(subject_path, header=None, names=["subject_id"])

print(subject_train.shape)
print(subject_train.head())

# %% Attach labels and subject IDs
feature_df['activity'] = y_train['activity_name'].values
feature_df['subject'] = subject_train['subject_id'].values

print(feature_df.shape)
feature_df.head()

# %% Quick sanity check: compare std feature across activities
feature_df.groupby('activity')['body_acc_x_std'].mean().sort_values()

# %% Save the feature table to disk
feature_df.to_csv("train_features.csv", index=False)
print("Saved.")


# %% Load all 6 raw signal axes for the TEST set
raw_signals_test = {}

for sig in signal_types:
    path = DATA_PATH + rf"\test\Inertial Signals\{sig}_test.txt"
    raw_signals_test[sig] = pd.read_csv(path, sep=r"\s+", header=None)
    print(f"{sig}: {raw_signals_test[sig].shape}")

# %% Load test labels and subject IDs
y_test_path = DATA_PATH + r"\test\y_test.txt"
y_test = pd.read_csv(y_test_path, header=None, names=["activity_id"])
y_test["activity_name"] = y_test["activity_id"].map(activity_map)

subject_test_path = DATA_PATH + r"\test\subject_test.txt"
subject_test = pd.read_csv(subject_test_path, header=None, names=["subject_id"])

print(y_test.shape, subject_test.shape)

# %% Build the feature table for the TEST set (same loop as before)
num_windows_test = raw_signals_test['body_acc_x'].shape[0]
all_features_test = []

for i in range(num_windows_test):
    row_features = {}
    for sig in signal_types:
        window = raw_signals_test[sig].iloc[i].values
        feats = extract_features(window)
        for key, value in feats.items():
            row_features[f"{sig}_{key}"] = value
    all_features_test.append(row_features)

    if i % 500 == 0:
        print(f"Processed {i}/{num_windows_test} windows")

feature_df_test = pd.DataFrame(all_features_test)
feature_df_test['activity'] = y_test['activity_name'].values
feature_df_test['subject'] = subject_test['subject_id'].values

print(feature_df_test.shape)

# %% Save test features too
feature_df_test.to_csv("test_features.csv", index=False)
print("Saved test features.")

# %% Prepare X (features) and y (labels) for training
feature_cols = [col for col in feature_df.columns if col not in ['activity', 'subject']]

X_train = feature_df[feature_cols]
y_train_labels = feature_df['activity']

X_test = feature_df_test[feature_cols]
y_test_labels = feature_df_test['activity']

print(X_train.shape, X_test.shape)

# %% Train a Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train_labels)

y_pred = clf.predict(X_test)

print(classification_report(y_test_labels, y_pred))

# %% Build and visualize the confusion matrix
import seaborn as sns

cm = confusion_matrix(y_test_labels, y_pred, labels=clf.classes_)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Activity Classification")
plt.tight_layout()
plt.savefig("sensor-activity-classifier/reports/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()


# %% Get feature importances from the trained model
importances = pd.Series(clf.feature_importances_, index=feature_cols)
importances_sorted = importances.sort_values(ascending=False)

print(importances_sorted.head(20))

# %% Plot top 20 most important features
plt.figure(figsize=(10, 8))
importances_sorted.head(20).sort_values().plot(kind='barh')
plt.xlabel("Feature Importance")
plt.title("Top 20 Most Important Features — Random Forest")
plt.tight_layout()
plt.savefig("sensor-activity-classifier/reports/feature_importance_overall.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Isolate standing and sitting rows only, train a focused mini-classifier
mask = feature_df['activity'].isin(['STANDING', 'SITTING'])
X_subset = feature_df.loc[mask, feature_cols]
y_subset = feature_df.loc[mask, 'activity']

clf_subset = RandomForestClassifier(n_estimators=200, random_state=42)
clf_subset.fit(X_subset, y_subset)

subset_importances = pd.Series(clf_subset.feature_importances_, index=feature_cols)
subset_importances_sorted = subset_importances.sort_values(ascending=False)

print(subset_importances_sorted.head(15))

# %% Visualize focused importance ranking
plt.figure(figsize=(10, 6))
subset_importances_sorted.head(15).sort_values().plot(kind='barh', color='orange')
plt.xlabel("Feature Importance")
plt.title("Top Features Specifically Separating STANDING vs SITTING")
plt.tight_layout()
plt.savefig("sensor-activity-classifier/reports/feature_importance_standing_vs_sitting.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Combine train and test features into one dataset (30 subjects)
full_df = pd.concat([feature_df, feature_df_test], ignore_index=True)

X = full_df[feature_cols]
y = full_df['activity']
groups = full_df['subject']

print("Total windows:", X.shape[0])
print("Unique subjects:", groups.nunique())

# %% Subject-wise 5-fold cross-validation
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score

gkf = GroupKFold(n_splits=5)

fold_acc, fold_f1 = [], []
all_true, all_pred = [], []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    # Safety check: no subject may appear in both train and test
    train_subjects = set(groups.iloc[train_idx])
    test_subjects = set(groups.iloc[test_idx])
    assert train_subjects.isdisjoint(test_subjects), "Subject leakage!"

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[test_idx])

    acc = accuracy_score(y.iloc[test_idx], pred)
    f1 = f1_score(y.iloc[test_idx], pred, average='macro')
    fold_acc.append(acc)
    fold_f1.append(f1)
    all_true.extend(y.iloc[test_idx])
    all_pred.extend(pred)

    print(f"Fold {fold+1}: test subjects={sorted(test_subjects)}  acc={acc:.3f}  macro-F1={f1:.3f}")

print()
print(f"Accuracy : {np.mean(fold_acc):.3f} ± {np.std(fold_acc):.3f}")
print(f"Macro-F1 : {np.mean(fold_f1):.3f} ± {np.std(fold_f1):.3f}")

# %% Compare: random (leaky) KFold vs subject-wise GroupKFold
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
leaky_acc = []

for train_idx, test_idx in kf.split(X):
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    leaky_acc.append(accuracy_score(y.iloc[test_idx], model.predict(X.iloc[test_idx])))

print(f"Random split accuracy       : {np.mean(leaky_acc):.3f} ± {np.std(leaky_acc):.3f}  (leaky)")
print(f"Subject-wise split accuracy : {np.mean(fold_acc):.3f} ± {np.std(fold_acc):.3f}  (honest)")

# %% Leave-One-Subject-Out: accuracy per subject
from sklearn.model_selection import LeaveOneGroupOut

logo = LeaveOneGroupOut()
subject_acc = {}

for train_idx, test_idx in logo.split(X, y, groups):
    subj = groups.iloc[test_idx].iloc[0]
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    subject_acc[subj] = accuracy_score(y.iloc[test_idx], model.predict(X.iloc[test_idx]))

subject_acc = pd.Series(subject_acc).sort_index()
print(subject_acc.describe())

plt.figure(figsize=(12, 4))
subject_acc.plot(kind='bar')
plt.xlabel("Held-out subject")
plt.ylabel("Accuracy")
plt.title("Leave-One-Subject-Out Accuracy per Subject")
plt.tight_layout()
plt.savefig("sensor-activity-classifier/reports/loso_per_subject_accuracy.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Confusion matrix aggregated across all CV folds
from sklearn.metrics import confusion_matrix
import seaborn as sns

labels = sorted(y.unique())
cm_cv = confusion_matrix(all_true, all_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_cv, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Subject-wise 5-Fold CV (all folds combined)")
plt.tight_layout()
plt.savefig("sensor-activity-classifier/reports/confusion_matrix_cv.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Imports for model comparison
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


# %% Encode labels and create ONE set of folds reused by every model
le = LabelEncoder()
y_enc = le.fit_transform(y)          # y from the full_df step; strings -> integers
class_names = list(le.classes_)
print(class_names)

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X, y_enc, groups))   # computed once, reused below
print("Number of folds:", len(splits))

# %% Define the models (Dummy first, as the "no learning" baseline)
models = {
    "Dummy (most frequent)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=42))
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, random_state=42))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        random_state=42, n_jobs=-1, eval_metric="mlogloss"),
}

# %% Evaluate every model on the same subject-wise folds
sit_id = class_names.index("SITTING")
stand_id = class_names.index("STANDING")

results = []
cv_predictions = {}

for name, model in models.items():
    accs, f1s, train_times, infer_times = [], [], [], []
    all_true, all_pred = [], []

    for train_idx, test_idx in splits:
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y_enc[train_idx], y_enc[test_idx]

        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        train_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        pred = model.predict(X_te)
        infer_times.append((time.perf_counter() - t0) / len(X_te) * 1000)  # ms per window

        accs.append(accuracy_score(y_te, pred))
        f1s.append(f1_score(y_te, pred, average="macro", zero_division=0))
        all_true.extend(y_te)
        all_pred.extend(pred)

    per_class_f1 = f1_score(all_true, all_pred, average=None, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": f"{np.mean(accs):.3f} ± {np.std(accs):.3f}",
        "Macro-F1": f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}",
        "SITTING F1": round(per_class_f1[sit_id], 3),
        "STANDING F1": round(per_class_f1[stand_id], 3),
        "Train time (s/fold)": round(np.mean(train_times), 2),
        "Inference (ms/window)": round(np.mean(infer_times), 4),
    })
    cv_predictions[name] = (all_true, all_pred)
    print(f"Finished {name}")

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# %% Save comparison table
results_df.to_csv("sensor-activity-classifier/reports/model_comparison.csv", index=False)
print(results_df.to_markdown(index=False))

# %% Bar chart: macro-F1 by model
plot_df = pd.DataFrame({
    "Model": [r["Model"] for r in results],
    "Macro-F1": [float(r["Macro-F1"].split(" ")[0]) for r in results],
})
plt.figure(figsize=(9, 4))
plt.bar(plot_df["Model"], plot_df["Macro-F1"])
plt.ylim(0, 1.0)
plt.ylabel("Macro-F1 (subject-wise 5-fold CV)")
plt.title("Model comparison")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("sensor-activity-classifier/reports/model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Load raw signals, train rows first then test rows (matches full_df order)
SIGNALS = ['body_acc_x', 'body_acc_y', 'body_acc_z',
           'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
           'total_acc_x', 'total_acc_y', 'total_acc_z']

raw = {}
for sig in SIGNALS:
    tr = pd.read_csv(DATA_PATH + rf"\train\Inertial Signals\{sig}_train.txt", sep=r"\s+", header=None).values
    te = pd.read_csv(DATA_PATH + rf"\test\Inertial Signals\{sig}_test.txt", sep=r"\s+", header=None).values
    raw[sig] = np.vstack([tr, te])

print(raw['total_acc_x'].shape)
assert raw['total_acc_x'].shape[0] == len(full_df), "Row count mismatch with full_df"

# %% Group 1: gravity orientation features from total_acc
def gravity_features(raw):
    g = np.stack([raw['total_acc_x'].mean(axis=1),
                  raw['total_acc_y'].mean(axis=1),
                  raw['total_acc_z'].mean(axis=1)], axis=1)       # window-average gravity vector
    norm = np.linalg.norm(g, axis=1, keepdims=True)
    angles = np.degrees(np.arccos(np.clip(g / norm, -1, 1)))      # angle between gravity and each axis
    return pd.DataFrame({
        'grav_x': g[:, 0], 'grav_y': g[:, 1], 'grav_z': g[:, 2],
        'tilt_x_deg': angles[:, 0], 'tilt_y_deg': angles[:, 1], 'tilt_z_deg': angles[:, 2],
    })

grav_df = gravity_features(raw)
print(grav_df.groupby(y.values).mean().round(3))

# %% Group 2: signal magnitude and axis correlation features
def row_corr(a, b):
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
    safe = np.where(den > 0, den, 1.0)
    return np.where(den > 0, (a * b).sum(axis=1) / safe, 0.0)

def magnitude_corr_features(raw):
    out = {}
    for name, prefix in [('acc', 'body_acc'), ('gyro', 'body_gyro')]:
        arr = np.stack([raw[f'{prefix}_{a}'] for a in 'xyz'], axis=2)   # (n, 128, 3)
        mag = np.linalg.norm(arr, axis=2)
        out[f'{name}_mag_mean'] = mag.mean(axis=1)
        out[f'{name}_mag_std'] = mag.std(axis=1)
        for i, j, lab in [(0, 1, 'xy'), (0, 2, 'xz'), (1, 2, 'yz')]:
            out[f'{name}_corr_{lab}'] = row_corr(arr[:, :, i], arr[:, :, j])
    return pd.DataFrame(out)

mag_df = magnitude_corr_features(raw)
print(mag_df.shape)

# %% Group 3: frequency features (Hz-aware)
FS = 50  # sampling rate in Hz
BODY_SIGNALS = ['body_acc_x', 'body_acc_y', 'body_acc_z',
                'body_gyro_x', 'body_gyro_y', 'body_gyro_z']

def frequency_features(raw):
    out = {}
    freqs = np.fft.rfftfreq(128, d=1 / FS)                 # frequency of each FFT bin in Hz
    bands = {'low_0.1-1Hz': (0.1, 1), 'mid_1-3Hz': (1, 3), 'high_3-10Hz': (3, 10)}
    for sig in BODY_SIGNALS:
        x = raw[sig] - raw[sig].mean(axis=1, keepdims=True)
        P = np.abs(np.fft.rfft(x, axis=1)) ** 2             # power spectrum
        total = P.sum(axis=1) + 1e-12
        p = P / total[:, None]

        out[f'{sig}_dom_freq_hz'] = freqs[P[:, 1:].argmax(axis=1) + 1]       # skip DC bin
        out[f'{sig}_spec_entropy'] = -(p * np.log2(p + 1e-12)).sum(axis=1) / np.log2(P.shape[1])
        for label, (lo, hi) in bands.items():
            m = (freqs >= lo) & (freqs < hi)
            out[f'{sig}_band_{label}'] = P[:, m].sum(axis=1) / total
    return pd.DataFrame(out)

freq_df = frequency_features(raw)
print(freq_df.shape)

# %% Ablation: add feature groups and measure the effect with XGBoost
from sklearn.metrics import confusion_matrix

def evaluate_config(name, feats):
    feats = np.nan_to_num(feats.reset_index(drop=True))
    feats = pd.DataFrame(feats)
    accs, f1s, all_true, all_pred = [], [], [], []
    for train_idx, test_idx in splits:
        model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                              random_state=42, n_jobs=-1, eval_metric="mlogloss")
        model.fit(feats.iloc[train_idx], y_enc[train_idx])
        pred = model.predict(feats.iloc[test_idx])
        accs.append(accuracy_score(y_enc[test_idx], pred))
        f1s.append(f1_score(y_enc[test_idx], pred, average="macro"))
        all_true.extend(y_enc[test_idx]); all_pred.extend(pred)

    per_class = f1_score(all_true, all_pred, average=None)
    cm = confusion_matrix(all_true, all_pred)
    return {
        "Config": name,
        "N features": feats.shape[1],
        "Accuracy": f"{np.mean(accs):.3f} ± {np.std(accs):.3f}",
        "Macro-F1": f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}",
        "SITTING F1": round(per_class[sit_id], 3),
        "STANDING F1": round(per_class[stand_id], 3),
        "Sit<->Stand errors": int(cm[sit_id, stand_id] + cm[stand_id, sit_id]),
    }

base = X.reset_index(drop=True)
configs = {
    "Baseline (66)": base,
    "+ Gravity": pd.concat([base, grav_df], axis=1),
    "+ Magnitude/Corr": pd.concat([base, mag_df], axis=1),
    "+ Frequency": pd.concat([base, freq_df], axis=1),
    "All new features": pd.concat([base, grav_df, mag_df, freq_df], axis=1),
}

ablation = []
for name, feats in configs.items():
    ablation.append(evaluate_config(name, feats))
    print("Finished", name)

ablation_df = pd.DataFrame(ablation)
print(ablation_df.to_markdown(index=False))

# %% Save ablation table
ablation_df.to_csv("sensor-activity-classifier/reports/feature_ablation.csv", index=False)

# %% CV predictions + per-fold macro-F1 for a feature set
def cv_predict(feats):
    feats = pd.DataFrame(np.nan_to_num(feats.reset_index(drop=True)))
    fold_f1, true, pred_all = [], [], []
    for train_idx, test_idx in splits:
        model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                              random_state=42, n_jobs=-1, eval_metric="mlogloss")
        model.fit(feats.iloc[train_idx], y_enc[train_idx])
        pred = model.predict(feats.iloc[test_idx])
        fold_f1.append(f1_score(y_enc[test_idx], pred, average="macro"))
        true.extend(y_enc[test_idx]); pred_all.extend(pred)
    return np.array(true), np.array(pred_all), np.array(fold_f1)

t_base, p_base, f_base = cv_predict(configs["Baseline (66)"])
t_all, p_all, f_all = cv_predict(configs["All new features"])

# Paired per-fold comparison (same folds for both)
diff = f_all - f_base
print("Per-fold macro-F1 gain:", np.round(diff, 3))
print(f"Mean gain {diff.mean():.3f}; improved in {(diff > 0).sum()}/5 folds")

# %% Side-by-side confusion matrices: baseline vs all features
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, (t, p, title) in zip(axes, [(t_base, p_base, "Baseline (66 features)"),
                                     (t_all, p_all, "All features (112)")]):
    sns.heatmap(confusion_matrix(t, p), annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False)
    ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig("sensor-activity-classifier/reports/confusion_baseline_vs_all.png", dpi=150, bbox_inches="tight")
plt.show()

