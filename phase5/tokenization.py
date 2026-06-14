import re
from collections import Counter

# ── 1. Simple word tokenization ───────────────────────
def word_tokenize(text):
    text = text.lower()
    tokens = re.findall(r'\b[a-zA-Z]+\b', text)
    return tokens

text = "I love building AI models! NLP is fascinating."
tokens = word_tokenize(text)
print("Word tokens:", tokens)

# ── 2. Build a vocabulary ─────────────────────────────
corpus = [
    "I love AI engineering",
    "AI models are powerful",
    "I love building models",
    "engineering is fascinating"
]

all_tokens = []
for sentence in corpus:
    all_tokens.extend(word_tokenize(sentence))

word_counts = Counter(all_tokens)
print("\nWord frequencies:", dict(word_counts))

# ── 3. Create word-to-index mapping ───────────────────
vocab = {"<PAD>": 0, "<UNK>": 1}  # special tokens
for word, count in word_counts.most_common():
    vocab[word] = len(vocab)

print(f"\nVocabulary size: {len(vocab)}")
print("Sample vocab:", dict(list(vocab.items())[:8]))

# ── 4. Encode sentences to numbers ────────────────────
def encode(sentence, vocab):
    tokens = word_tokenize(sentence)
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

def decode(indices, vocab):
    idx_to_word = {idx: word for word, idx in vocab.items()}
    return [idx_to_word.get(idx, "<UNK>") for idx in indices]

test_sentence = "I love AI engineering"
encoded = encode(test_sentence, vocab)
decoded = decode(encoded, vocab)

print(f"\nOriginal:  {test_sentence}")
print(f"Encoded:   {encoded}")
print(f"Decoded:   {decoded}")

# ── 5. Handle unknown words ───────────────────────────
unknown_sentence = "I love quantum computing"
encoded_unk = encode(unknown_sentence, vocab)
print(f"\nSentence with unknown word: {unknown_sentence}")
print(f"Encoded (UNK=1 for unknowns): {encoded_unk}")

# ── 6. Padding — making sequences same length ─────────
def pad_sequence(sequence, max_len, pad_token=0):
    if len(sequence) < max_len:
        sequence = sequence + [pad_token] * (max_len - len(sequence))
    return sequence[:max_len]

sentences = ["I love AI", "AI models are powerful tools", "NLP"]
encoded_sentences = [encode(s, vocab) for s in sentences]
max_len = max(len(s) for s in encoded_sentences)

padded = [pad_sequence(s, max_len) for s in encoded_sentences]
print("\n--- Padding demo ---")
for i, (orig, pad) in enumerate(zip(encoded_sentences, padded)):
    print(f"Sentence {i+1}: {orig} → padded: {pad}")
