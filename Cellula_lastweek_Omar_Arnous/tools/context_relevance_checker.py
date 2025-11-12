from langchain.prompts import PromptTemplate
import os
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
import os, sys
from dotenv import load_dotenv
from langchain.chains import LLMChain

load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "context_relevance_checker.txt")
with open(prompt_path, "r", encoding="utf-8") as f:
    relevance_prompt_text = f.read()

relevance_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=relevance_prompt_text
)

relevance_llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
    model="mistralai/mistral-7b-instruct:free",  #"deepseek/deepseek-chat-v3.1:free",
    temperature=0.0
)


relevance_chain = LLMChain(llm=relevance_llm, prompt=relevance_prompt)

def relevance_checker_function(tool_input: str) -> str:
    """
    Parses the input string 'query|context' and runs the relevance chain.
    """
    try:
        query, context = tool_input.split("|", 1)
    except ValueError:
        return "Error: Input must be in the format 'query|context'"
    
    return relevance_chain.run(query=query.strip(), context=context.strip())

ContextRelevanceTool = Tool(
    name="context_relevance_checker",
    description="**MANDATORY TOOL.** Before answering any question that involves external information or context (like a search or retrieval result), you MUST use this tool first. The input MUST be a single string in the format: `original user query|retrieved context string`. Its output is the ONLY context you can use to formulate your final answer. Determines if a given context is relevant to the user's query. Use it after each context retrieval to filter out irrelevant information.",
    func=relevance_checker_function 
)