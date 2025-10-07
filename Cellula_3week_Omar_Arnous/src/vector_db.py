import numpy as np
import faiss
from copy import deepcopy
from huggingface_hub import InferenceClient
import json
import os
from dotenv import load_dotenv


load_dotenv()
class VectorDB:
    def __init__(self, embedding_model_id="BAAI/bge-code-v1", data_path="helpers"):
        self.embedding_model_id = embedding_model_id
        self.hf_client = InferenceClient(api_key=os.getenv("HF_TOKEN"))
        self.data_path = data_path
        self._load_data()
        self._build_index()

    def _load_data(self):
        with open(os.path.join(self.data_path, "humaneval_processed.json"), "r") as f:
            self.processed_data = json.load(f)

        self.task_ids = [str(i) for i in range(len(self.processed_data))]
        self.prompts = [item["prompt"] for item in self.processed_data]
        self.solutions = [item["canonical_solution"] for item in self.processed_data]

        self.prompts_embeddings = np.load(os.path.join(self.data_path, "humaneval_embeddings.npy"))
        self.dimension = self.prompts_embeddings.shape[1]

    def _build_index(self):
        embeddings_norm = deepcopy(self.prompts_embeddings)
        faiss.normalize_L2(embeddings_norm)
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))
        self.index.add_with_ids(embeddings_norm, np.array(self.task_ids, dtype=np.int64))

    def search(self, query, top_k=3):
        query_embedding = self.hf_client.feature_extraction(query, model=self.embedding_model_id)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        scores, ids = self.index.search(query_embedding, top_k)
        return [int(i) for i in ids[0]]
