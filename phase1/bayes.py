import numpy as np

# ── 1. Bayes Theorem function ─────────────────────────
def bayes(prior, likelihood, evidence):
    return (likelihood * prior) / evidence

# ── 2. Spam detection example ────────────────────────
prior_spam = 0.3
likelihood_free_given_spam = 0.8
prob_free = 0.4

posterior = bayes(prior_spam, likelihood_free_given_spam, prob_free)
print(f"P(spam | FREE) = {posterior:.2f}")
print(f"Belief updated from {prior_spam} to {posterior:.2f}")

# ── 3. Medical diagnosis example ─────────────────────
# Disease affects 1% of population
# Test is 99% accurate
# You test positive — what's the real probability you have it?

prior_disease = 0.01          # 1% have disease
likelihood_pos_given_disease = 0.99   # test catches 99% of cases
likelihood_pos_given_no_disease = 0.01  # 1% false positive rate

# P(positive) = P(pos|disease)×P(disease) + P(pos|no disease)×P(no disease)
prob_positive = (likelihood_pos_given_disease * prior_disease +
                 likelihood_pos_given_no_disease * (1 - prior_disease))

posterior_disease = bayes(prior_disease,
                          likelihood_pos_given_disease,
                          prob_positive)

print(f"\nMedical test example:")
print(f"P(disease | positive test) = {posterior_disease:.3f}")
print(f"That's only {posterior_disease*100:.1f}% — not 99%!")

# ── 4. Naive Bayes classifier from scratch ───────────
print("\n--- Naive Bayes Classifier ---")

# Training data: emails with word counts
# Features: contains_free, contains_win, contains_meet
# Label: 1=spam, 0=not spam
emails = [
    {"free": 1, "win": 1, "meet": 0, "label": 1},
    {"free": 1, "win": 0, "meet": 0, "label": 1},
    {"free": 0, "win": 1, "meet": 0, "label": 1},
    {"free": 0, "win": 0, "meet": 1, "label": 0},
    {"free": 0, "win": 0, "meet": 1, "label": 0},
    {"free": 1, "win": 0, "meet": 1, "label": 0},
]

spam = [e for e in emails if e["label"] == 1]
not_spam = [e for e in emails if e["label"] == 0]

p_spam = len(spam) / len(emails)
p_not_spam = len(not_spam) / len(emails)

print(f"P(spam) = {p_spam:.2f}")
print(f"P(not spam) = {p_not_spam:.2f}")

# Likelihoods
for word in ["free", "win", "meet"]:
    p_word_spam = sum(e[word] for e in spam) / len(spam)
    p_word_not_spam = sum(e[word] for e in not_spam) / len(not_spam)
    print(f"P({word}|spam)={p_word_spam:.2f}  P({word}|not spam)={p_word_not_spam:.2f}")

# Classify new email: contains free=1, win=0, meet=0
print("\nClassifying new email: free=1, win=0, meet=0")
p_spam_score = p_spam * (2/3) * (1/3) * (1/3)
p_not_spam_score = p_not_spam * (1/3) * (0/3 + 0.01) * (2/3)
print(f"Spam score: {p_spam_score:.4f}")
print(f"Not spam score: {p_not_spam_score:.4f}")
print(f"Classification: {'SPAM' if p_spam_score > p_not_spam_score else 'NOT SPAM'}")