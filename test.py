# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# %% Visualize this focused importance ranking
plt.figure(figsize=(10, 6))
subset_importances_sorted.head(15).sort_values().plot(kind='barh', color='orange')
plt.xlabel("Feature Importance")
plt.title("Top Features Specifically Separating STANDING vs SITTING")
plt.tight_layout()
plt.show()