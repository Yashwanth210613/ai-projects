import re

# ── 1. Tools ────────────────────────────────────────────
def calculator(expression):
    try:
        allowed = "0123456789+-*/(). "
        if all(c in allowed for c in expression):
            return eval(expression)
        return "Error: invalid expression"
    except Exception as e:
        return f"Error: {e}"

def search_database(query):
    db = {
        "q3 revenue": 50000,
        "q2 revenue": 42000,
        "employee count": 150,
        "office locations": 3,
    }
    query_lower = query.lower()
    for key, value in db.items():
        if key in query_lower:
            return value
    return "Not found"

TOOLS = {"calculator": calculator, "search_database": search_database}

# ── 2. Multi-step reasoning simulation ────────────────
def simulate_multistep_reasoning(query, step, previous_observations):
    """
    Simulates an LLM reasoning across MULTIPLE steps,
    using results from previous steps to decide next action.
    """
    query_lower = query.lower()

    # Step 1: if query mentions a database fact AND math, first get the fact
    if step == 1:
        if "revenue" in query_lower or "employee" in query_lower or "office" in query_lower:
            return {"thought": "I need to look up a fact from the database first",
                    "tool": "search_database", "args": {"query": query}}

    # Step 2: use the previous observation in a calculation
    if step == 2 and previous_observations:
        last_value = previous_observations[-1]
        if "%" in query or "percent" in query_lower:
            percent_match = re.search(r'(\d+)\s*%', query)
            if percent_match:
                percent = int(percent_match.group(1))
                expression = f"{last_value} * {percent} / 100"
                return {"thought": f"Now I'll calculate {percent}% of {last_value}",
                        "tool": "calculator", "args": {"expression": expression}}

    return None

# ── 3. Full multi-step agent ───────────────────────────
def run_multistep_agent(query, max_steps=4):
    print(f"\n{'='*60}")
    print(f"User Query: {query}")
    print(f"{'='*60}")

    observations = []

    for step in range(1, max_steps + 1):
        print(f"\n--- Step {step} ---")
        decision = simulate_multistep_reasoning(query, step, observations)

        if decision is None:
            if observations:
                print(f"Thought: I have enough information now.")
                print(f"Final Answer: {observations[-1]}")
            else:
                print("Thought: No tool needed.")
                print(f"Final Answer: I understand: {query}")
            return

        print(f"Thought: {decision['thought']}")
        print(f"Action: {decision['tool']}({decision['args']})")

        tool_fn = TOOLS[decision['tool']]
        result = tool_fn(**decision['args'])
        observations.append(result)

        print(f"Observation: {result}")

# ── 4. Test multi-step reasoning ───────────────────────
test_queries = [
    "What is 15% of our Q3 revenue?",
    "What is 20% of our Q2 revenue?",
]

for query in test_queries:
    run_multistep_agent(query)
