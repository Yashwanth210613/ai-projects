import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

np.random.seed(42)

# ── 1. Generate data ──────────────────────────────────
X, y = make_classification(
    n_samples=1000, n_features=4, n_informative=3,
    n_redundant=1, random_state=42
)
feature_names = ["age", "income", "credit_score", "debt_ratio"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 2. Single Decision Tree — shallow vs deep ────────
print("--- Decision Tree: Depth Comparison ---")
for depth in [2, 5, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, tree.predict(X_train))
    test_acc  = accuracy_score(y_test, tree.predict(X_test))

    depth_label = depth if depth else "Unlimited"
    print(f"Max Depth {str(depth_label):10s} | "
          f"Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")

# ── 3. Print the actual tree (depth=2) ───────────────
tree_shallow = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_shallow.fit(X_train, y_train)
print("\n--- Tree Structure (depth=2) ---")
print(export_text(tree_shallow, feature_names=feature_names))

# ── 4. Feature importance ─────────────────────────────
print("--- Feature Importance (Single Tree) ---")
for name, importance in zip(feature_names, tree_shallow.feature_importances_):
    print(f"{name:15s}: {importance:.3f}")

# ── 5. Random Forest comparison ───────────────────────
print("\n--- Single Tree vs Random Forest ---")

single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)
tree_acc = accuracy_score(y_test, single_tree.predict(X_test))

forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)
forest_acc = accuracy_score(y_test, forest.predict(X_test))

print(f"Single Tree Accuracy:  {tree_acc:.3f}")
print(f"Random Forest Accuracy: {forest_acc:.3f}")

print("\n--- Feature Importance (Random Forest) ---")
for name, importance in zip(feature_names, forest.feature_importances_):
    print(f"{name:15s}: {importance:.3f}")

# ── 6. Cross-validation ───────────────────────────────
print("\n--- Cross-Validation (5-fold) ---")
scores = cross_val_score(forest, X_train, y_train, cv=5)
print(f"Fold scores: {scores.round(3)}")
print(f"Mean: {scores.mean():.3f} | Std: {scores.std():.3f}")
