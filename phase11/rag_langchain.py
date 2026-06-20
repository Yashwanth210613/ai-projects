from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

print("=== LangChain RAG Pipeline ===\n")

# ── 1. Sample document — a longer text to chunk ───────
long_document = """
Company Policy Document

Returns and Refunds:
Our return policy allows customers to return items within 30 days of purchase
for a full refund. Items must be in original condition with tags attached.
Refunds are processed within 5-7 business days after we receive the returned item.

Shipping Information:
Shipping typically takes 3-5 business days for domestic orders and 7-14 days
for international orders. We offer free shipping on orders over $50 within
the United States. Express shipping is available for an additional fee.

Customer Support:
Our customer support team is available Monday to Friday, 9 AM to 6 PM EST.
You can reach us via email, live chat, or phone. Premium members get priority
support with response times under 1 hour.

Payment Methods:
We accept payments via credit card, PayPal, and Apple Pay. All transactions
are secured with industry-standard encryption. Gift cards never expire and
can be used for any purchase on our website.

Exchanges:
Products can be exchanged for a different size or color within 14 days of
delivery. Exchange shipping is free for the first exchange per order.
"""

# ── 2. Chunking with LangChain's text splitter ────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,       # characters per chunk
    chunk_overlap=30,     # overlap between chunks (preserves context)
    separators=["\n\n", "\n", ". ", " "]
)

chunks = text_splitter.split_text(long_document)
print(f"Document split into {len(chunks)} chunks\n")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} ({len(chunk)} chars):")
    print(f"  {chunk.strip()}\n")

# ── 3. Convert chunks to LangChain Documents ──────────
documents = [Document(page_content=chunk) for chunk in chunks]

# ── 4. Create embeddings and vector store ─────────────
print("Creating vector store...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents, embeddings)

print("Vector store created with ChromaDB\n")

# ── 5. Retrieval with LangChain ────────────────────────
print("=== Retrieval Test ===")
queries = [
    "How do I get my money back?",
    "What are your support hours?",
    "Can I pay with PayPal?",
]

for query in queries:
    print(f"\nQuery: '{query}'")
    results = vectorstore.similarity_search_with_score(query, k=2)
    for doc, score in results:
        print(f"  [score={score:.3f}] {doc.page_content.strip()[:80]}...")

# ── 6. Build retriever for RAG chain ──────────────────
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
print("\n=== Retriever ready for RAG chain ===")
print("This retriever can now be plugged into any LLM chain")
print("(OpenAI, Anthropic, HuggingFace models, etc.)")
