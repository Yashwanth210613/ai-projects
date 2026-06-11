
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# ── 1. Sigmoid function ───────────────────────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print("--- Sigmoid Demo ---")
for x in [-5, -2, 0, 2, 5]:
    print(f"sigmoid({x:2d}) = {sigmoid(x):.3f}")

# ── 2. Generate classification data ──────────────────
# Predicting if student passes (1) or fails (0)
# Features: study_hours, sleep_hours
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=42
)

# ── 3. Split and scale ────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 4. Train model ────────────────────────────────────
model = LogisticRegression()
model.fit(X_train, y_train)

# ── 5. Predict ────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ── 6. Evaluation metrics ─────────────────────────────
print("\n--- Model Evaluation ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.3f}")

# ── 7. Confusion matrix ───────────────────────────────
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"TN={cm[0][0]}  FP={cm[0][1]}")
print(f"FN={cm[1][0]}  TP={cm[1][1]}")

# ── 8. Classification report ──────────────────────────
print("\nFull Report:")
print(classification_report(y_test, y_pred))

# ── 9. What happens at different thresholds ───────────
print("--- Threshold Impact ---")
for threshold in [0.3, 0.5, 0.7]:
    y_thresh = (y_prob >= threshold).astype(int)
    print(f"Threshold {threshold} | "
          f"Precision: {precision_score(y_test, y_thresh):.3f} | "
          f"Recall: {recall_score(y_test, y_thresh):.3f}")
