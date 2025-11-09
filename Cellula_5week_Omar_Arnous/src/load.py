import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from datasets import load_dataset

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def get_llm(model_name): 
    llm = ChatOpenAI(
    model=model_name,
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0.0
    )
    return llm 

dataset = load_dataset("openai/openai_humaneval", split="test")
df = dataset.to_pandas()

subset = df[["task_id", "prompt", "canonical_solution"]]
subset.columns
print(subset.head())