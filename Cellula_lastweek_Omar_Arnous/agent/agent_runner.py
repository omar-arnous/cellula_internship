import os
import sys
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langsmith import Client
from langchain_core.messages import AIMessage

# ---- Import Tools ----
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tools.context_presence_judge import *
from tools.web_search_tool import *
from tools.context_relevance_checker import *
from tools.context_splitter import *

# ---- Setup LangSmith (Optional for tracing) ----
client = Client()
client.create_project(f"Agent-Debug-Project-{int(time.time())}")

# ---- Load Env Vars ----
load_dotenv()
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# ---- Initialize LLM ----
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
    model="gpt-4o-mini",
    temperature=0.25,
    max_tokens=2048
).with_config({"verbose": True})

# ---- Initialize Tools ----
ContextPresenceJudge = context_presence_tool(llm)
ContextSplitter = build_context_splitter_tool(llm)

tools = [
    ContextPresenceJudge,
    WebSearchTool,
    ContextRelevanceTool,
    ContextSplitter
]

# ---- Memory ----
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ---- ReAct Agent ----
react_agent = create_react_agent(llm, tools)

# ---- Optional Cleanup Chain ----
template = """
You are an AI research assistant. Use the following information to answer clearly and concisely:
{context}

Now, summarize the key point as a clean final answer:
"""
prompt = PromptTemplate(input_variables=["context"], template=template)
cleanup_chain = LLMChain(llm=llm, prompt=prompt)

# ---- Helper Function ----
def extract_last_ai_message(response):
    if isinstance(response, dict) and "messages" in response:
        messages = response["messages"]
    else:
        messages = response if isinstance(response, list) else [response]

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content
        elif isinstance(msg, dict) and msg.get("role") == "assistant":
            return msg.get("content")

    return str(response)

# ---- FastAPI App ----
app = FastAPI(title="Cellula AI Agent", version="1.0")

class UserMessage(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "🧠 Cellula AI Agent is running successfully!"}

@app.post("/chat")
def chat_with_agent(user_input: UserMessage):
    try:
        # Combine memory + user input
        past_convo = "\n".join([f"{m.type.capitalize()}: {m.content}" for m in memory.chat_memory.messages])
        full_input = f"{past_convo}\nUser: {user_input.message}"

        # Invoke Agent
        response = react_agent.invoke(
            {"messages": [{"role": "user", "content": full_input}]},
            config={"recursion_limit": 50, "verbose": True, "run_name": "Chat Session"}
        )

        clean_output = extract_last_ai_message(response)

        # Update memory
        memory.chat_memory.add_user_message(user_input.message)
        memory.chat_memory.add_ai_message(clean_output)

        return {"response": clean_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))