import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
openrouter_api_key = os.getenv("LLM_ID")
embedding_model_id = "BAAI/bge-large-en-v1.5"

class HFInferenceEmbedding:
    def __init__(self, model_name, api_key):
        self.client = InferenceClient(provider="hf-inference", api_key=api_key)
        self.model_name = model_name
    
    def embed_documents(self, texts):
        return [self.client.feature_extraction(text, model=self.model_name) for text in texts]
    
    def embed_query(self, text):
        return self.client.feature_extraction(text, model=self.model_name)
    
# Load Chroma vectorstore
vectorstore = Chroma(
    collection_name="cellula_collection",
    embedding_function=HFInferenceEmbedding(model_name=embedding_model_id, api_key=hf_token),
    persist_directory="../data/chroma_db"
)
print("✅ Chroma DB loaded successfully!")

# OpenRouter LLM (Changed to ChatOpenAI)
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
    model="meta-llama/llama-3.3-8b-instruct:free", #"mistralai/mistral-7b-instruct:free", "meta-llama/llama-3.3-8b-instruct:free"
    temperature=0.25,
    max_tokens=2048
)

# Define the prompt template with rules
prompt = ChatPromptTemplate.from_messages([
   ("system", """
        You are an AI Customer Support Assistant that answers user inquiries based on the rules below.

        Four Rules for generating answers:
        1. Context Usage:
        Use the provided context (retrieved documents) only if the question is related to products, services, policies, troubleshooting, or customer account information.
        Do not mention, hint, or imply that you have access to retrieved documents or any external data source.

        2. When Context Is Missing Information:
        If the question is related to customer support but the answer is not explicitly stated in the provided context, reply exactly with:
        "Unfortunately, I don’t have the information needed to assist with this request."

        3. Non-Support or General Questions:
        If the question is not related to customer support (e.g., casual, personal, or unrelated queries), ignore the context and answer normally using your general knowledge.

        4. Response Formatting:
        Always provide a textual answer (never leave the response empty).
        Do not include any special tokens (e.g., <s>, </s>, [INST], etc.).
        Keep your responses polite, empathetic, clear, and concise.

        The Tasks that you can do:
        1. Answer questions about products, services, pricing, policies, and procedures.
        2. Provide troubleshooting steps or guidance using the retrieved context.
        3. Respond to general, non-support questions in a friendly and professional manner.

        ### Context (retrieved documents):
        {context}
    """),       
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Per-user chat history storage
chat_histories = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Get or create chat history for a specific session/user.
    This follows the new LangChain pattern.
    """
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]

def get_session_history_with_window(session_id: str, k: int = 5):
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    history = chat_histories[session_id]
    # Truncate messages if too long
    if len(history.messages) > 2 * k:
        history.messages = history.messages[-2*k:]
    return history

# Create the chain with memory
def create_chain_with_history():
    """
    Creates a runnable chain that includes message history.
    """
    # Create chain: prompt | llm | output parser
    chain = prompt | llm | StrOutputParser()
    
    # Wrap it with message history
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history_with_window,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    
    return chain_with_history

# Exposed function to answer questions
def answer_question(question: str, chat_id: str = "default"):
    """
    Answer a question using RAG with per-user chat history.
    
    Args:
        question: The user's question
        chat_id: Unique identifier for the chat session (default: "default")
    
    Returns:
        tuple: (answer, similar_docs)
    """
    # 1. Retrieve similar docs from vectorstore
    similar_docs = vectorstore.similarity_search(question, k=4)
    context = "---A Retrieved Document---"+"\n\n---A Retrieved Document---\n".join([doc.page_content for doc in similar_docs])

    # 2. Create chain with history
    chain_with_history = create_chain_with_history()
    
    # 3. Invoke with session config
    config = {"configurable": {"session_id": chat_id}}
    
    # 4. Get response
    answer = chain_with_history.invoke(
        {
            "context": context,
            "question": question
        },
        config=config
    )
    print(f"Debug: Initial answer: {answer}")
    while answer.strip() == "":
        print("⚠️ Empty answer received, retrying...")
        # clear history and try again
        clear_history(chat_id)
        # repeat to get answer
        answer = chain_with_history.invoke(
            {
                "context": context,
                "question": question
            },
            config=config
        )
    print(f"✅✅✅✅ Answer generated: {answer}")

    # print the chat history to txt file for debugging
    chat_history = get_session_history(chat_id)
    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file_path = os.path.join(LOG_DIR, f"retrieved_docs_log({chat_id}).txt")
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(f"\nChat history for session {chat_id}:\n")
        for message in chat_history.messages:
            f.write(f"{message}\n")
        f.write(f"{'-'*100}\n")

    return answer, similar_docs


# Optional: Clear history for a specific user
def clear_history(chat_id: str):
    """Clear chat history for a specific user/session."""
    if chat_id in chat_histories:
        del chat_histories[chat_id]
        print(f"✅ History cleared for session: {chat_id}")

__all__ = ["answer_question", "clear_history"]