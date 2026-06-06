import numpy as np

# ── 1. Creating vectors ──────────────────────────────
v1 = np.array([2, 3, 5])
v2 = np.array([1, 4, 2])

print("v1:", v1)
print("v2:", v2)

# ── 2. Vector operations ─────────────────────────────
print("\nAddition:", v1 + v2)
print("Subtraction:", v1 - v2)
print("Scalar multiply:", v1 * 3)

# ── 3. Dot product ───────────────────────────────────
# Measures similarity between two vectors
dot = np.dot(v1, v2)
print("\nDot product:", dot)

# ── 4. Vector magnitude (length) ─────────────────────
magnitude = np.linalg.norm(v1)
print("Magnitude of v1:", round(magnitude, 2))

# ── 5. Matrix operations ─────────────────────────────
M1 = np.array([[1, 2], [3, 4]])
M2 = np.array([[5, 6], [7, 8]])

print("\nMatrix M1:\n", M1)
print("Matrix M2:\n", M2)
print("Matrix addition:\n", M1 + M2)
print("Matrix multiply:\n", np.matmul(M1, M2))

# ── 6. Transpose ─────────────────────────────────────
print("\nTranspose of M1:\n", M1.T)

# ── 7. Real AI use case — cosine similarity ──────────
# How similar are two word vectors?
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

word1 = np.array([0.8, 0.3, 0.1])  # "king"
word2 = np.array([0.7, 0.4, 0.2])  # "queen"
word3 = np.array([0.1, 0.9, 0.8])  # "banana"

print("\nSimilarity king-queen:", round(cosine_similarity(word1, word2), 3))
print("Similarity king-banana:", round(cosine_similarity(word1, word3), 3))