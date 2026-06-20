from transformers import GPT2Tokenizer, AutoTokenizer
import torch

# ── 1. Load GPT-2 tokenizer (same BPE as GPT-3/4) ────
print("=== GPT-2 BPE Tokenizer ===")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
print(f"Vocabulary size: {tokenizer.vocab_size:,}")

# ── 2. See how BPE handles different text ─────────────
texts = [
    "Hello world",
    "I love artificial intelligence",
    "def fibonacci(n): return n if n<=1 else fibonacci(n-1)+fibonacci(n-2)",
    "unconstitutionally",
    "ChatGPT is amazing!!!",
    "மிகவும் நன்றி",  # Tamil text
]

print("\n--- Tokenization Examples ---")
for text in texts:
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)
    print(f"\nText:   {text}")
    print(f"Tokens: {tokens}")
    print(f"Count:  {len(tokens)} tokens")

# ── 3. The token count matters for cost ───────────────
print("\n=== Token Count & Cost ===")
long_text = """
Machine learning is a subset of artificial intelligence that enables
systems to learn and improve from experience without being explicitly
programmed. It focuses on developing computer programs that can access
data and use it to learn for themselves.
"""
tokens = tokenizer.encode(long_text)
print(f"Text length: {len(long_text)} characters")
print(f"Token count: {len(tokens)} tokens")
print(f"Ratio: {len(long_text)/len(tokens):.1f} chars per token")
print(f"\nOpenAI GPT-4 pricing: ~$0.01 per 1K tokens")
print(f"This text would cost: ${len(tokens) * 0.00001:.6f}")

# ── 4. Special tokens ─────────────────────────────────
print("\n=== Special Tokens ===")
print(f"EOS token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")
print(f"BOS token: '{tokenizer.bos_token}' (id: {tokenizer.bos_token_id})")

# ── 5. Encoding & decoding ────────────────────────────
print("\n=== Encode → Decode Round Trip ===")
original = "Transformers changed everything in NLP"
encoded = tokenizer.encode(original)
decoded = tokenizer.decode(encoded)
print(f"Original: {original}")
print(f"Encoded:  {encoded}")
print(f"Decoded:  {decoded}")
print(f"Perfect reconstruction: {original == decoded}")

# ── 6. Context window demonstration ───────────────────
print("\n=== Context Window ===")
max_context = 1024  # GPT-2's context window
print(f"GPT-2 context window: {max_context} tokens")
print(f"GPT-4 context window: 128,000 tokens")
print(f"Claude context window: 200,000 tokens")
print(f"\nAt ~0.75 words/token:")
print(f"GPT-2 can 'see': ~{int(max_context*0.75):,} words at once")
print(f"GPT-4 can 'see': ~{int(128000*0.75):,} words at once")
print(f"Claude can 'see': ~{int(200000*0.75):,} words at once")
