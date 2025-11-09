from typing import Dict, Any, List
from load import get_llm
from prepare import vectordb 
from langchain_core.documents import Document
from typing import Dict, Any

RETRIEVE_TOP_K = 5
CODE_PROMPT_TEMPLATE = """
You are a code generation assistant. Given the user's request, produce a single, final Python implementation.
Constraints:
- Output only valid Python code. No explanation, no comments outside the code.
- If examples are provided, use them only if they are relevant; otherwise ignore them.
- Keep to idiomatic python and include necessary imports.

User request:
---
{user_input}
---
{examples_block}
"""

def retrieve_similar_examples(user_input: str, top_k: int = RETRIEVE_TOP_K) -> List[Document]:
    # Use the in-memory vectordb to get the most similar examples
    results = vectordb.similarity_search(user_input, k=top_k)
    return results

def build_code_prompt(user_input: str, examples: List[Document]) -> str:
    if not examples:
        examples_block = ""
    else:
        items = []
        for ex in examples:
            prompt = ex.metadata.get("prompt", "")
            sol = ex.metadata.get("canonical_solution", "")
            items.append(f"### Example prompt:\n{prompt}\n### Example solution:\n{sol}\n")
        examples_block = "\n\n".join(items)
    return CODE_PROMPT_TEMPLATE.format(user_input=user_input, examples_block=examples_block)

def generate_code(state: Dict[str, Any], model_name: str = "gpt-4o-mini") -> Dict[str, Any]:
    # Extract the actual user input string from the state
    user_input = state["user_input"]
    
    # Initialize the LLM
    llm = get_llm(model_name=model_name)
    
    # Retrieve similar examples based on the user input
    examples = retrieve_similar_examples(user_input)
    
    # Build the prompt
    prompt = build_code_prompt(user_input, examples)
    
    # Invoke the model
    response = llm.invoke(prompt)
    final_code = response.content.strip()
    
    # Save outputs back into the state
    state["code_output"] = final_code
    state["prompt_used"] = prompt
    state["num_examples"] = len(examples)
    
    # Return updated state (LangGraph expects the state dict)
    return state