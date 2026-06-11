import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(42)

# ── 1. Generate data ──────────────────────────────────
# Predicting house price based on size
size = np.random.randint(500, 3000, 100).reshape(-1, 1)
price = 150 * size.flatten() + np.random.randn(100) * 20000 + 50000

# ── 2. Split data ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    size, price, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ── 3. Train model ────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

print(f"\nLearned weight (slope): {model.coef_[0]:.2f}")
print(f"Learned bias:           {model.intercept_:.2f}")

# ── 4. Evaluate ───────────────────────────────────────
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"\nMSE:  {mse:,.0f}")
print(f"RMSE: {np.sqrt(mse):,.0f}")
print(f"R²:   {r2:.3f}")

# ── 5. Predict a new house ────────────────────────────
new_house = np.array([[1500]])
predicted_price = model.predict(new_house)
print(f"\nPredicted price for 1500 sqft: ${predicted_price[0]:,.0f}")

# ── 6. Demonstrate overfitting ────────────────────────
print("\n--- Overfitting Demo ---")
degrees = [1, 3, 10, 20]

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly  = poly.transform(X_test)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)

    train_r2 = r2_score(y_train, poly_model.predict(X_train_poly))
    test_r2  = r2_score(y_test,  poly_model.predict(X_test_poly))

    status = "✓ Good" if abs(train_r2 - test_r2) < 0.05 else "✗ Overfit"
    print(f"Degree {degree:2d} | Train R²: {train_r2:.3f} | "
          f"Test R²: {test_r2:.3f} | {status}")