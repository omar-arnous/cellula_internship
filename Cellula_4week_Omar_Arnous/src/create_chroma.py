from huggingface_hub import InferenceClient
from langchain_chroma import Chroma
import json
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
openrouter_api_key = os.getenv("LLM_ID")

embedding_model_id = "BAAI/bge-large-en-v1.5"

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(DATA_DIR, exist_ok=True)

# I created this class to interface with Hugging Face Inference API for embeddings, to be used with Chroma
class HFInferenceEmbeddings:
    def __init__(self, model_name: str, api_key: str):
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=api_key
        )
        self.model_name = model_name

    def embed_documents(self, texts):
        """Embeds a list of texts (for Chroma)."""
        return [self.client.feature_extraction(text, model=self.model_name) for text in texts]

    def embed_query(self, text):
        """Embeds a single query string (for retrieval)."""
        return self.client.feature_extraction(text, model=self.model_name)

def create_chroma_db():
    # Load your precomputed embeddings
    embeddings_path = os.path.join(DATA_DIR, "wikipedia_embeddings_bge_base_v1.5.json")
    with open(embeddings_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} documents with embeddings.")

    # Prepare Chroma inputs
    texts = [item["content"] for item in data]
    metadatas = [item.get("metadata", {}) for item in data]
    embeddings = [np.array(item["embedding"], dtype=np.float32) for item in data]
    ids = [str(i) for i in range(len(data))]

    # Setup persist directory
    persist_directory = os.path.join(DATA_DIR, "chroma_db")
    os.makedirs(persist_directory, exist_ok=True)

    # Initialize embedding model (for future queries)
    embedding_function = HFInferenceEmbeddings(
        model_name=embedding_model_id,
        api_key=hf_token
    )

    # Create a Chroma store
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )

    # Add precomputed embeddings manually
    vectorstore._collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✅ Stored {len(embeddings)} precomputed embeddings in {persist_directory}")
    print(f'the name of collection: {vectorstore._collection.name}')