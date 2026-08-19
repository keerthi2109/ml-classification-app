"""
Train all 5 classification models on the Wine Quality dataset (Red + White combined).
Binary classification: Quality >= 7 is 'Good' (1), else 'Not Good' (0).
Features: 11 physicochemical properties + 1 color indicator = 12 features.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

# --- Step 1: Load and Prepare Dataset ---
print("Loading Wine Quality dataset...")

# Download datasets
red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
white_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

red_wine = pd.read_csv(red_wine_url, sep=';')
white_wine = pd.read_csv(white_wine_url, sep=';')

# Add color feature: red=1, white=0
red_wine['color'] = 1
white_wine['color'] = 0

# Combine datasets
wine_data = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)

# Binary target: Quality >= 7 is 'Good' (1), else 'Not Good' (0)
wine_data['target'] = (wine_data['quality'] >= 7).astype(int)
wine_data = wine_data.drop('quality', axis=1)

print(f"Dataset shape: {wine_data.shape}")
print(f"Number of features: {wine_data.shape[1] - 1}")
print(f"Number of instances: {wine_data.shape[0]}")
print(f"Class distribution:\n{wine_data['target'].value_counts()}")

# --- Step 2: Split features and target ---
X = wine_data.drop('target', axis=1)
y = wine_data['target']

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save test data for Streamlit app
test_data = X_test.copy()
test_data['target'] = y_test.values
test_data.to_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data.csv'), index=False)
print("\nTest data saved to test_data.csv")

# --- Step 3: Define and Train Models ---
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# --- Step 4: Train, Evaluate, and Save Models ---
results = {}

print("\n" + "=" * 80)
print("MODEL TRAINING AND EVALUATION")
print("=" * 80)

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    results[name] = {
        'Accuracy': round(accuracy, 4),
        'AUC': round(auc, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1 Score': round(f1, 4),
        'MCC': round(mcc, 4)
    }

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  MCC: {mcc:.4f}")

# --- Step 5: Save models, scaler, and results ---
model_dir = os.path.dirname(__file__)

# Save each model
for name, model in models.items():
    filename = name.lower().replace(' ', '_') + '.pkl'
    filepath = os.path.join(model_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved {name} to {filename}")

# Save scaler
with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
print("Saved scaler.pkl")

# Save results
with open(os.path.join(model_dir, 'results.pkl'), 'wb') as f:
    pickle.dump(results, f)
print("Saved results.pkl")

# Save feature names
with open(os.path.join(model_dir, 'feature_names.pkl'), 'wb') as f:
    pickle.dump(list(X.columns), f)
print("Saved feature_names.pkl")

# --- Step 6: Print Comparison Table ---
print("\n" + "=" * 80)
print("MODEL COMPARISON TABLE")
print("=" * 80)
results_df = pd.DataFrame(results).T
print(results_df.to_string())

# Find overall winner
best_model = results_df['F1 Score'].idxmax()
print(f"\nOverall Winner (by F1 Score): {best_model}")
print(f"  with F1 Score: {results_df.loc[best_model, 'F1 Score']}")
