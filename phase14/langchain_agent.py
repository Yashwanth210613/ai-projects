from langchain_core.tools import Tool

print("=== LangChain Agent Framework Structure ===\n")

# ── 1. Define tools using LangChain's Tool class ──────
def calculator_tool(expression: str) -> str:
    try:
        allowed = "0123456789+-*/(). "
        if all(c in allowed for c in expression):
            return str(eval(expression))
        return "Invalid expression"
    except Exception as e:
        return str(e)

def knowledge_base_tool(query: str) -> str:
    kb = {
        "return": "Items can be returned within 30 days.",
        "shipping": "Free shipping over $50.",
    }
    for key, val in kb.items():
        if key in query.lower():
            return val
    return "No info found."

tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Useful for math calculations. Input should be a math expression."
    ),
    Tool(
        name="KnowledgeBase",
        func=knowledge_base_tool,
        description="Useful for answering questions about company policies like returns and shipping."
    ),
]

print("Tools registered:")
for tool in tools:
    print(f"  - {tool.name}: {tool.description}")

# ── 2. Test calling tools directly ────────────────────
print("\n=== Testing Tools ===")
result1 = tools[0].func("15 * 100 / 100")
print(f"Calculator('15 * 100 / 100') = {result1}")

result2 = tools[1].func("what is your return policy")
print(f"KnowledgeBase('return policy') = {result2}")

# ── 3. This is the EXACT structure production agents use ──
print("\n=== Production Agent Structure (for reference) ===")
print("""
In production, this is how you'd connect a REAL LLM:

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

llm = ChatOpenAI(model="gpt-4", api_key="your-key")
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": "What is 15% of 50000?"})
""")

print("=== Key Takeaway ===")
print("The TOOLS definition is identical whether you use GPT-4, Claude,")
print("or any LLM — only the 'brain' (LLM) and AgentExecutor wiring changes.")
print("You've now built this exact tool-calling pattern 3 times:")
print("  1. From scratch (manual)")
print("  2. Multi-step reasoning (manual)")
print("  3. LangChain's Tool class (production-standard)")
