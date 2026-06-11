# mini_project.py

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

data = load_breast_cancer()

X = data.data
y = data.target

print("=" * 60)
print("Dataset Loaded Successfully")
print("Dataset Shape:", X.shape)
print("Target Classes:", data.target_names)
print("=" * 60)

# --------------------------------------------------
# 2. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain Size:", len(X_train))
print("Test Size :", len(X_test))

# --------------------------------------------------
# 3. Models
# --------------------------------------------------

log_model = LogisticRegression(max_iter=10000)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# --------------------------------------------------
# 4. Train Models
# --------------------------------------------------

log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# --------------------------------------------------
# 5. Predictions
# --------------------------------------------------

log_pred = log_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# --------------------------------------------------
# 6. Evaluation Function
# --------------------------------------------------

def evaluate_model(name, y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{name}")
    print("-" * 40)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return accuracy, precision, recall, f1


log_metrics = evaluate_model(
    "Logistic Regression",
    y_test,
    log_pred
)

rf_metrics = evaluate_model(
    "Random Forest",
    y_test,
    rf_pred
)

# --------------------------------------------------
# 7. Cross Validation
# --------------------------------------------------

print("\n" + "=" * 60)
print("5-Fold Cross Validation")
print("=" * 60)

log_cv = cross_val_score(
    log_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

rf_cv = cross_val_score(
    rf_model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nLogistic Regression CV Accuracy:")
print(log_cv)
print("Average:", log_cv.mean())

print("\nRandom Forest CV Accuracy:")
print(rf_cv)
print("Average:", rf_cv.mean())

# --------------------------------------------------
# 8. Determine Best Model
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

if rf_cv.mean() > log_cv.mean():
    print("Best Model: Random Forest")
    print(
        "Reason: Higher average cross-validation accuracy, "
        "better ability to capture complex patterns."
    )
else:
    print("Best Model: Logistic Regression")
    print(
        "Reason: Higher average cross-validation accuracy "
        "and simpler model with better interpretability."
    )

print("\nProject Completed Successfully!")
