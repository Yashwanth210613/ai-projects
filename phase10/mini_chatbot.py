from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

torch.manual_seed(42)

print("Loading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()
tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, max_new_tokens=40, temperature=0.8):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3  # reduces repetition
        )

    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    response = full_text[len(prompt):].strip()
    return response

# ── Test prompts ───────────────────────────────────────
test_prompts = [
    "The key benefits of machine learning are",
    "When building an AI project, the first step is",
    "Python is a popular programming language because",
]

print("\n=== Mini Chatbot Responses ===\n")
for prompt in test_prompts:
    response = generate_response(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}\n")
    print("-" * 60)

# ── Token usage tracking (like real API costs) ────────
print("\n=== Token Usage Summary ===")
total_tokens = 0
for prompt in test_prompts:
    tokens = len(tokenizer.encode(prompt))
    total_tokens += tokens

print(f"Total prompt tokens: {total_tokens}")
print(f"Estimated cost (GPT-4 pricing): ${total_tokens * 0.00003:.5f}")
