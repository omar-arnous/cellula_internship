from huggingface_hub import InferenceClient
import numpy as np
import json
from tqdm import tqdm
import os
import time
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
openrouter_api_key = os.getenv("LLM_ID")

# embedding_model_id = "BAAI/bge-base-en-v1.5"
embedding_model_id = "BAAI/bge-large-en-v1.5"

# Initialize the Hugging Face inference client
hf_client = InferenceClient(
    provider="hf-inference",
    api_key=hf_token,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(DATA_DIR, exist_ok=True)

def get_embedding_with_retry(text, model, retries=10, wait=2):
    """
    Try to get embedding multiple times in case of server errors.
    Retries on HTTP 5xx, connection errors, or timeouts.
    """
    for attempt in range(retries):
        try:
            # Try to get embedding
            emb = hf_client.feature_extraction(text, model=model)
            return np.array(emb, dtype=np.float32)
        
        except (requests.exceptions.RequestException, Exception) as e:
            print(f"⚠️ Error on attempt {attempt+1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(wait * (attempt + 1))
            else:
                print(f"❌ Failed after {retries} retries. Skipping this text.")
                return None
            
def embed_documents():
    # Load the pre-split documents
    chunks_path = os.path.join(DATA_DIR, "wikipedia_docs_chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} chunks.")

    # Embed all documents safely
    embeddings = []
    failed_indices = []

    for i, item in enumerate(tqdm(data, desc="Embedding chunks")):
        text = item["content"].strip()
        emb = get_embedding_with_retry(text, embedding_model_id)
        if emb is not None:
            embeddings.append(emb)
        else:
            failed_indices.append(i)

    print(f"✅ Finished embedding. {len(failed_indices)} items failed.")
    print(f"✅ Created {len(embeddings)} embeddings of dimension {embeddings[0].shape[0]}")

    # Combine text, metadata, and embeddings
    embedded_data = [
        {
            "content": item["content"],
            "metadata": item["metadata"],
            "embedding": emb.tolist()
        }
        for item, emb in zip(data, embeddings)
    ]

    # Save to file
    embeddings_path = os.path.join(DATA_DIR, "wikipedia_embeddings_bge_base_v1.5.json")
    with open(embeddings_path, "w", encoding="utf-8") as f:
        json.dump(embedded_data, f, ensure_ascii=False, indent=4)

    print("💾 Saved all embeddings to wikipedia_embeddings_bge_base_v1.5.json")