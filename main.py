import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print(f"Key loaded: {api_key[:10]}...")