import torch
import torch.nn as nn
import numpy as np

# ── 1. PyTorch Embedding layer ────────────────────────
# vocab_size=10, embedding_dim=4
embedding = nn.Embedding(num_embeddings=10, embedding_dim=4)

print("--- Embedding Layer ---")
print(f"Vocabulary size: 10")
print(f"Embedding dimension: 4")
print(f"Total parameters: {10 * 4} (each word gets 4 numbers)")

word_indices = torch.tensor([2, 3, 4])  # "i", "love", "ai"
embedded = embedding(word_indices)
print(f"\nInput indices: {word_indices.tolist()}")
print(f"Output embeddings shape: {embedded.shape}")
print(f"Embeddings:\n{embedded.detach()}")

# ── 2. Embeddings are LEARNED during training ─────────
print("\n--- Embeddings are trained parameters ---")
print(f"requires_grad: {embedding.weight.requires_grad}")
print("These numbers start random and get updated via backprop")
print("After training, similar words will have similar vectors")

# ── 3. Simulate Word2Vec-style similarity ─────────────
print("\n--- Simulating Word2Vec similarity ---")
np.random.seed(42)

# Pre-trained-style word vectors (simplified)
word_vectors = {
    "king":   np.array([0.8, 0.3, 0.9, 0.1]),
    "queen":  np.array([0.7, 0.4, 0.8, 0.2]),
    "man":    np.array([0.6, 0.1, 0.4, 0.3]),
    "woman":  np.array([0.5, 0.2, 0.3, 0.4]),
    "apple":  np.array([0.1, 0.9, 0.1, 0.8]),
    "orange": np.array([0.2, 0.8, 0.2, 0.7]),
    "paris":  np.array([0.9, 0.1, 0.5, 0.6]),
    "france": np.array([0.8, 0.2, 0.6, 0.5]),
}

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def most_similar(word, word_vectors, top_n=3):
    target = word_vectors[word]
    sims = {w: cosine_similarity(target, v)
            for w, v in word_vectors.items() if w != word}
    return sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_n]

print("\nMost similar to 'king':")
for word, sim in most_similar("king", word_vectors):
    print(f"  {word}: {sim:.3f}")

print("\nMost similar to 'apple':")
for word, sim in most_similar("apple", word_vectors):
    print(f"  {word}: {sim:.3f}")

# ── 4. Word2Vec arithmetic ────────────────────────────
print("\n--- Word2Vec arithmetic ---")
result = word_vectors["king"] - word_vectors["man"] + word_vectors["woman"]
sims = {w: cosine_similarity(result, v) for w, v in word_vectors.items()}
closest = max(sims, key=sims.get)
print(f"king - man + woman = ?")
print(f"Closest word: '{closest}' (similarity: {sims[closest]:.3f})")

# ── 5. OOV problem demonstration ──────────────────────
print("\n--- OOV (Out of Vocabulary) problem ---")
vocab_size = len(word_vectors)
embed_dim = 4
embed_layer = nn.Embedding(vocab_size, embed_dim)

print(f"Vocabulary: {list(word_vectors.keys())}")
print("If we see 'emperor' at inference time → NOT in vocabulary → UNK")
print("BPE solution: 'emperor' → ['emp', 'eror'] → both subwords in vocab")
