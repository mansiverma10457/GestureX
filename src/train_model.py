import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

dataset_path = "dataset"

data = []
labels = []

print("Loading dataset...")

# -------------------------------
# Load all gesture CSV files
# -------------------------------
for gesture in os.listdir(dataset_path):

    gesture_folder = os.path.join(dataset_path, gesture)

    if not os.path.isdir(gesture_folder):
        continue

    for file in os.listdir(gesture_folder):

        file_path = os.path.join(gesture_folder, file)

        df = pd.read_csv(file_path, header=None)

        data.append(df.values.flatten())
        labels.append(gesture)

# Convert to DataFrame
X = pd.DataFrame(data)
y = pd.Series(labels)

print("Dataset loaded")
print("Samples:", len(X))
print("Features:", X.shape[1])
print("Gestures:", y.unique())

# -------------------------------
# Shuffle dataset
# -------------------------------
dataset = pd.concat([X, y], axis=1)
dataset = dataset.sample(frac=1).reset_index(drop=True)

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# -------------------------------
# Train / Test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# -------------------------------
# Train RandomForest model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

print("Training model...")

model.fit(X_train, y_train)

# -------------------------------
# Test accuracy
# -------------------------------
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nTest Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# -------------------------------
# Cross Validation (Real Accuracy)
# -------------------------------
scores = cross_val_score(model, X, y, cv=5)

print("\nCross Validation Scores:", scores)
print("Average CV Accuracy:", scores.mean())

# -------------------------------
# Save trained model
# -------------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/gesture_model.pkl")

print("\nModel saved to models/gesture_model.pkl")