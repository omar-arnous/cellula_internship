from typing import Dict, Any
from dotenv import load_dotenv
from load import get_llm
import os, json
from langchain_openai import ChatOpenAI

# --- Prompt Template ---
INTENT_PROMPT = """
You are an AI assistant whose sole task is to classify the user's request into one of two intents: "generate_code" or "explain_code".
Return *only* a valid JSON object with one key: "intent".

User query:
---
{user_input}
---

Instructions:
- If the user asks for writing, creating, or generating code in any programming language → {{ "intent": "generate_code" }}
- If the user asks to explain, analyze, or describe what a piece of code does → {{ "intent": "explain_code" }}

Examples:
1. "Write a Python function that reverses a string" → {{ "intent": "generate_code" }}
2. "Explain what this code does: def foo(): ..." → {{ "intent": "explain_code" }}

Now, classify the user's request:
"""

llm = get_llm("meta-llama/llama-3.1-8b-instruct")

def classify_intent(state):
    """
    Classify user intent using the LLM from OpenRouter.
    """
    user_input = state["user_input"] 
    prompt = INTENT_PROMPT.format(user_input=user_input.strip())
    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    import re 
    try:
        cleaned = (
            raw.strip()
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # fallback: extract "intent" manually
        match = re.search(r'"intent"\s*:\s*"(\w+)"', raw)
        if match:
            parsed = {"intent": match.group(1)}
        else:
            parsed = {"intent": "unknown"}


    return {"intent": parsed["intent"], "raw": raw}

# # --- Test the function ---
# if __name__ == "__main__":
#     test_inputs = [
#         "Write a Python function that reverses a string",
#         "Explain what this code does: for i in range(5): print(i)"
#     ]

#     for inp in test_inputs:
#         print(f"\nUser input: {inp}")
#         result = classify_intent(inp)
#         print("Result:", json.dumps(result, indent=2))