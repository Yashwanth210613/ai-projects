from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ── 1. Our "knowledge base" — company documents ───────
documents = [
    "Our return policy allows customers to return items within 30 days of purchase for a full refund.",
    "Shipping typically takes 3-5 business days for domestic orders and 7-14 days for international orders.",
    "We offer free shipping on orders over $50 within the United States.",
    "Our customer support team is available Monday to Friday, 9 AM to 6 PM EST.",
    "Premium membership costs $9.99 per month and includes free shipping on all orders.",
    "Products can be exchanged for a different size within 14 days of delivery.",
    "We accept payments via credit card, PayPal, and Apple Pay.",
    "Gift cards never expire and can be used for any purchase on our website.",
]

print(f"Knowledge base: {len(documents)} documents")

# ── 2. Embed all documents (INDEXING phase) ───────────
print("\nEmbedding documents...")
doc_embeddings = embedder.encode(documents)
print(f"Embedding shape: {doc_embeddings.shape}")
print(f"({len(documents)} documents, {doc_embeddings.shape[1]} dimensions each)")

# ── 3. Cosine similarity search function ──────────────
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, documents, doc_embeddings, top_k=2):
    query_embedding = embedder.encode([query])[0]

    similarities = [
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in doc_embeddings
    ]

    # Get indices of top-k most similar documents
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = [(documents[i], similarities[i]) for i in top_indices]
    return results

# ── 4. Test retrieval ──────────────────────────────────
print("\n=== RAG Retrieval Demo ===")
queries = [
    "How long do I have to return something?",
    "Do you ship internationally?",
    "What payment methods do you accept?",
]

for query in queries:
    print(f"\nQuery: '{query}'")
    results = retrieve(query, documents, doc_embeddings, top_k=2)
    for doc, score in results:
        print(f"  [{score:.3f}] {doc}")

# ── 5. Full RAG — retrieval + generation context ──────
print("\n=== Building the Final Prompt for LLM ===")
query = "How long do I have to return something?"
results = retrieve(query, documents, doc_embeddings, top_k=2)

retrieved_context = "\n".join([doc for doc, score in results])

prompt = f"""Answer the question based only on the following context:

Context:
{retrieved_context}

Question: {query}

Answer:"""

print(prompt)
print("\n--- This prompt would now be sent to GPT/Claude/any LLM ---")
print("--- The LLM answers using ONLY the retrieved context, not its own training knowledge ---")
