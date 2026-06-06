import numpy as np
import matplotlib.pyplot as plt

# ── 1. Basic transformation ───────────────────────────
M = np.array([[2, 1],
              [1, 3]])

v = np.array([1, 1])
transformed = M @ v
print("Original vector:", v)
print("Transformed vector:", transformed)

# ── 2. Eigenvalues & Eigenvectors ────────────────────
eigenvalues, eigenvectors = np.linalg.eig(M)

print("\nEigenvalues:", eigenvalues.round(2))
print("Eigenvectors:\n", eigenvectors.round(2))

# ── 3. Verify: M·v = λ·v ────────────────────────────
for i in range(len(eigenvalues)):
    lam = eigenvalues[i]
    vec = eigenvectors[:, i]
    lhs = M @ vec
    rhs = lam * vec
    print(f"\nEigenvector {i+1}: {vec.round(3)}")
    print(f"M·v = {lhs.round(3)}")
    print(f"λ·v = {rhs.round(3)}")
    print(f"Match: {np.allclose(lhs, rhs)}")

# ── 4. Real use case — PCA from scratch ─────────────
np.random.seed(42)
# Fake dataset: 100 students, 2 features (math score, physics score)
data = np.random.multivariate_normal(
    mean=[70, 75],
    cov=[[100, 80], [80, 100]],
    size=100
)

# Step 1: Center the data
data_centered = data - data.mean(axis=0)

# Step 2: Covariance matrix
cov_matrix = np.cov(data_centered.T)
print("\nCovariance matrix:\n", cov_matrix.round(2))

# Step 3: Eigenvalues of covariance matrix
vals, vecs = np.linalg.eig(cov_matrix)
print("\nPCA Eigenvalues:", vals.round(2))
print("Variance explained:", (vals / vals.sum() * 100).round(1), "%")

# Step 4: Project onto top eigenvector (reduce 2D → 1D)
top_vec = vecs[:, np.argmax(vals)]
projected = data_centered @ top_vec
print("\nOriginal shape:", data.shape)
print("Projected shape:", projected.shape)
print("We kept", round(max(vals)/sum(vals)*100, 1), "% of variance")