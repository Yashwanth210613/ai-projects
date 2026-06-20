import json
import re
import math
from datetime import datetime

# ── 1. Define real tools the agent can use ────────────
def calculator(expression):
    """Safely evaluate a math expression"""
    try:
        # Only allow safe math operations
        allowed = "0123456789+-*/(). "
        if all(c in allowed for c in expression):
            return str(eval(expression))
        return "Error: invalid expression"
    except Exception as e:
        return f"Error: {e}"

def get_current_time():
    """Get the current time"""
    return datetime.now().strftime("%H:%M:%S")

def search_knowledge_base(query):
    """Simulated company knowledge base search"""
    knowledge = {
        "return policy": "Items can be returned within 30 days for a full refund.",
        "shipping": "Free shipping on orders over $50, takes 3-5 business days.",
        "support hours": "Customer support available Monday-Friday, 9 AM to 6 PM EST.",
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower or any(word in query_lower for word in key.split()):
            return value
    return "No information found in knowledge base."

# ── 2. Tool registry — maps names to functions ────────
TOOLS = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "search_knowledge_base": search_knowledge_base,
}

TOOL_DESCRIPTIONS = """
Available tools:
1. calculator(expression: str) - Evaluates math expressions like "5+3*2"
2. get_current_time() - Returns the current time
3. search_knowledge_base(query: str) - Searches company policies

To use a tool, respond EXACTLY in this format:
TOOL_CALL: {"tool": "tool_name", "args": {"param": "value"}}

If you don't need a tool, respond normally with FINAL_ANSWER: your response
"""

# ── 3. Simple rule-based "LLM brain" (simulating decision-making) ──
def simulate_llm_decision(user_query):
    """
    In a real system, this would be GPT-4/Claude deciding which tool to use.
    Here we simulate that decision logic to understand the MECHANISM.
    """
    query_lower = user_query.lower()

    # Math detection
    math_pattern = re.search(r'[\d+\-*/().\s]+', user_query)
    if any(op in user_query for op in ['+', '-', '*', '/']) and any(c.isdigit() for c in user_query):
        expression = re.sub(r'[^\d+\-*/(). ]', '', user_query)
        return {"tool": "calculator", "args": {"expression": expression.strip()}}

    # Time detection
    if "time" in query_lower and "what" in query_lower:
        return {"tool": "get_current_time", "args": {}}

    # Knowledge base detection
    if any(word in query_lower for word in ["return", "shipping", "support", "policy"]):
        return {"tool": "search_knowledge_base", "args": {"query": user_query}}

    return None  # no tool needed

# ── 4. The Agent Loop ──────────────────────────────────
def run_agent(user_query, max_steps=3):
    print(f"\n{'='*60}")
    print(f"User Query: {user_query}")
    print(f"{'='*60}")

    for step in range(max_steps):
        print(f"\n--- Step {step+1} ---")
        print("Thought: Analyzing the query to decide if a tool is needed...")

        decision = simulate_llm_decision(user_query)

        if decision is None:
            print("Thought: No tool needed, I can answer directly.")
            print(f"Final Answer: I understand your question: '{user_query}'")
            return

        tool_name = decision["tool"]
        tool_args = decision["args"]

        print(f"Action: Calling tool '{tool_name}' with args {tool_args}")

        # Execute the actual tool
        tool_function = TOOLS[tool_name]
        result = tool_function(**tool_args)

        print(f"Observation: {result}")
        print(f"Final Answer: Based on the tool result, the answer is: {result}")
        return

# ── 5. Test the agent ──────────────────────────────────
test_queries = [
    "What is 847 + 392?",
    "What is your return policy?",
    "What time is it right now?",
    "Tell me about your shipping options",
]

for query in test_queries:
    run_agent(query)
