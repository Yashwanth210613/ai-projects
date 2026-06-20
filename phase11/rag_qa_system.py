from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

print("=== Full RAG Q&A System ===\n")

# ── 1. Knowledge base ──────────────────────────────────
company_docs = """
Return Policy:
Customers can return any item within 30 days of purchase for a full refund.
The item must be unused and in its original packaging. Refunds are processed
within 5-7 business days after we receive the returned item.

Shipping Policy:
We offer free standard shipping on all orders over $50. Standard shipping
takes 3-5 business days. Express shipping is available for $15 and takes
1-2 business days. We currently ship only within the United States.

Account & Billing:
You can update your payment method anytime in Account Settings. We accept
Visa, Mastercard, American Express, and PayPal. Subscriptions auto-renew
monthly unless cancelled at least 24 hours before the renewal date.

Technical Support:
If you experience login issues, try resetting your password using the
"Forgot Password" link. For persistent issues, contact support at
help@company.com with your account email and a description of the problem.
"""

# ── 2. Chunk and embed ─────────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40)
chunks = splitter.split_text(company_docs)
documents = [Document(page_content=c) for c in chunks]

print(f"Created {len(documents)} chunks")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ── 3. Load LLM for generation ─────────────────────────
print("\nLoading GPT-2 for generation...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

def generate_answer(prompt, max_new_tokens=50):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3
        )
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return full_text[len(prompt):].strip()

# ── 4. Full RAG pipeline function ──────────────────────
def rag_answer(question):
    # RETRIEVAL
    retrieved_docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # AUGMENTED PROMPT
    prompt = f"""Context: {context}

Question: {question}
Answer:"""

    # GENERATION
    answer = generate_answer(prompt)
    return answer, retrieved_docs

# ── 5. Test the full system ────────────────────────────
print("\n=== RAG Q&A System Live Test ===\n")
test_questions = [
    "How long do I have to return an item?",
    "Do you offer free shipping?",
    "What should I do if I can't log in?",
]

for question in test_questions:
    print(f"Q: {question}")
    answer, docs = rag_answer(question)
    print(f"Retrieved context: {docs[0].page_content[:60].strip()}...")
    print(f"A: {answer}\n")
    print("-" * 60)
