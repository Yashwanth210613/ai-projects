from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

torch.manual_seed(42)

print("Loading GPT-2 (small open-source LLM)...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

params = sum(p.numel() for p in model.parameters())
print(f"GPT-2 parameters: {params:,}")
print(f"(GPT-4 is estimated ~1.7 TRILLION — GPT-2 is tiny by comparison)")

# ── 1. Basic generation ───────────────────────────────
print("\n=== Basic Text Generation ===")
prompt = "Artificial intelligence will"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=30,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Generated: {generated_text}")

# ── 2. Temperature — controlling randomness ───────────
print("\n=== Temperature Comparison ===")
prompt = "The future of technology is"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

for temp in [0.3, 0.7, 1.5]:
    torch.manual_seed(42)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=25,
            do_sample=True,
            temperature=temp,
            pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\nTemperature {temp}:")
    print(f"  {text}")

# ── 3. Top-k and Top-p sampling ───────────────────────
print("\n=== Sampling Strategies ===")
prompt = "My favorite programming language is"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

strategies = [
    {"name": "Greedy (deterministic)", "do_sample": False},
    {"name": "Top-k=50", "do_sample": True, "top_k": 50, "temperature": 0.8},
    {"name": "Top-p=0.9 (nucleus)", "do_sample": True, "top_p": 0.9, "temperature": 0.8},
]

for strategy in strategies:
    torch.manual_seed(42)
    name = strategy.pop("name")
    with torch.no_grad():
        output = model.generate(
            input_ids, max_length=25,
            pad_token_id=tokenizer.eos_token_id,
            **strategy
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n{name}:")
    print(f"  {text}")

# ── 4. Manual greedy decoding — see the mechanism ─────
print("\n=== Manual Greedy Decoding (see autoregressive loop) ===")
prompt = "I think machine learning is"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

print(f"Starting prompt: {prompt}\n")
for step in range(5):
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]  # logits for next token

    next_token_id = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
    next_token_text = tokenizer.decode(next_token_id[0])

    print(f"Step {step+1}: predicted token = '{next_token_text}'")

    input_ids = torch.cat([input_ids, next_token_id], dim=1)

final_text = tokenizer.decode(input_ids[0])
print(f"\nFinal generated text: {final_text}")
