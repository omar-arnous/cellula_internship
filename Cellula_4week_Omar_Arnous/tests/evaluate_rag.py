import math
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

############################### Load environment variables ####################################
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
embedding_model_id = "BAAI/bge-large-en-v1.5"

############################### Define embedding model and vectorstore ####################################
# Hugging Face embeddings wrapper
class HFInferenceEmbeddings:
    def __init__(self, model_name, api_key):
        self.client = InferenceClient(provider="hf-inference", api_key=api_key)
        self.model_name = model_name
    
    def embed_documents(self, texts):
        return [self.client.feature_extraction(text, model=self.model_name) for text in texts]
    
    def embed_query(self, text):
        return self.client.feature_extraction(text, model=self.model_name)    

# Load Chroma vectorstore
vectorstore = Chroma(
    persist_directory="../data/chroma_db",
    embedding_function=HFInferenceEmbeddings(model_name=embedding_model_id, api_key=hf_token)
)
print("✅ Chroma DB loaded successfully!")


############################### Precision@K and Recall@K Functions ####################################

def precision_at_k(retrieved_docs, expected_docs, k):
    if k == 0:
        return 0.0
    
    # Take only top k retrieved documents
    top_k_docs = retrieved_docs[:k]
    
    # Count how many of the top k are in expected docs
    relevant_count = sum(1 for doc in top_k_docs if doc in expected_docs)
    
    return relevant_count / k


def recall_at_k(retrieved_docs, expected_docs, k):
    if len(expected_docs) == 0:
        return 0.0
    
    # Take only top k retrieved documents
    top_k_docs = retrieved_docs[:k]
    
    # Count how many of the top k are in expected docs
    relevant_count = sum(1 for doc in top_k_docs if doc in expected_docs)
    
    return relevant_count / len(expected_docs)


def dcg(relevances):
    """Compute Discounted Cumulative Gain."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    # Assign relevance scores (1 if relevant, else 0)
    relevances = [1 if doc in relevant else 0 for doc in retrieved_k]
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg(ideal_relevances)
    return dcg(relevances) / idcg if idcg > 0 else 0.0


def evaluate_retrieval(retrieved_docs, expected_docs, k_values=[1, 3, 5, 10]):
    results = {}
    
    for k in k_values:
        precision = precision_at_k(retrieved_docs, expected_docs, k)
        recall = recall_at_k(retrieved_docs, expected_docs, k)
        ndcg = ndcg_at_k(retrieved_docs, expected_docs, k)
        
        results[f"precision@{k}"] = precision
        results[f"recall@{k}"] = recall
        results[f"ndcg@{k}"] = ndcg
    
    return results


############################### Example Usage ####################################

query = """A deterministic stationary policy deterministically selects actions based on the current state.
the United States Copyright Office (USCO) released extensive guidance regarding the use of AI tools in the creative process.
A ChatGPT search involves the use of 10 times the electrical energy as a Google search.
"""

expected_context = [
    "A deterministic stationary policy deterministically selects actions based on the current state.\n\nSince any such policy can be identified with a mapping from the set of states to the set of actions, these policies can be identified with such mappings with no loss of generality.\n\n=== Brute force ===\nThe brute force approach entails two steps:\n\nFor each possible policy, sample returns while following it\nChoose the policy with the largest expected discounted return\nOne problem with this is that the number of policies can be large, or even infinite.\n\nAnother is that the variance of the returns may be large, which requires many samples to accurately estimate the discounted return of each policy.",
    "In January 2025, the United States Copyright Office (USCO) released extensive guidance regarding the use of AI tools in the creative process, and established that \"...generative AI systems also offer tools that similarly allow users to exert control.\n\n[These] can enable the user to control the selection and placement of individual creative elements.\n\nWhether such modifications rise to the minimum standard of originality required under Feist will depend on a case-by-case determination.\n\nIn those cases where they do, the output should be copyrightable\" Subsequently, the USCO registered the first visual artwork to be composed of entirely AI-generated materials, titled \"A Single Piece of American Cheese\".",
    "A ChatGPT search involves the use of 10 times the electrical energy as a Google search.\n\nThe large firms are in haste to find power sources – from nuclear energy to geothermal to fusion.\n\nThe tech firms argue that – in the long view – AI will be eventually kinder to the environment, but they need the energy now.\n\nAI makes the power grid more efficient and \"intelligent\", will assist in the growth of nuclear power, and track overall carbon emissions, according to technology firms.",
]

# Perform retrieval
retrieved_results = vectorstore.similarity_search(query, k=10)
retrieved_docs = [doc.page_content for doc in retrieved_results]

# Evaluate at different K values
evaluation_results = evaluate_retrieval(retrieved_docs, expected_context, k_values=[1, 3, 5, 10])

# Print results
print("\n" + "="*60)
print("RETRIEVAL EVALUATION RESULTS")
print("="*60)
for metric, score in evaluation_results.items():
    print(f"{metric:15s}: {score:.4f} ({score*100:.2f}%)")