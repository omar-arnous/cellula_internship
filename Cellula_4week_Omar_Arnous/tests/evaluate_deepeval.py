from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import ContextualRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.metrics import ContextualRelevancyMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from deepeval import assert_test, evaluate
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient
from deepeval import evaluate
from deepeval import assert_test
import requests
import json
from pydantic import BaseModel


#################################### Load environment variables ####################################
from dotenv import load_dotenv
import os
load_dotenv()
openrouter_api_key = os.getenv("LLM_ID")
CONFIDENT_API_KEY = os.getenv("CONFIDENT_API_KEY")
os.environ["CONFIDENT_API_KEY"] = CONFIDENT_API_KEY
hf_token = os.getenv("HF_TOKEN")
embedding_model_id = "BAAI/bge-large-en-v1.5"


######################################## Define  custom model ########################################
class OpenRouterModel(DeepEvalBaseLLM):
    def __init__(self, model="meta-llama/llama-3.2-3b-instruct:free", api_key=None):
        self.model = model
        self.api_key = api_key
        
    def load_model(self):
        return self.model
    
    def generate(self, prompt: str, schema=None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://deepeval-test.com",
            "X-Title": "DeepEval Test"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        # If schema is needed, request JSON in the prompt
        if schema:
            prompt_with_schema = f"{prompt}\n\nPlease respond with valid JSON only, no other text."
            data["messages"] = [{"role": "user", "content": prompt_with_schema}]
        
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code != 200:
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.text}")
            
            response.raise_for_status()
            response_json = response.json()
            
            if "choices" not in response_json:
                print("API Response:", response_json)
                raise KeyError(f"Unexpected API response structure: {response_json}")
            
            content = response_json["choices"][0]["message"]["content"]
            
            # If schema is provided, parse JSON and return as Pydantic model
            if schema:
                try:
                    # Clean the response - remove markdown code blocks if present
                    content_cleaned = content.strip()
                    if content_cleaned.startswith("```json"):
                        content_cleaned = content_cleaned[7:]
                    if content_cleaned.startswith("```"):
                        content_cleaned = content_cleaned[3:]
                    if content_cleaned.endswith("```"):
                        content_cleaned = content_cleaned[:-3]
                    content_cleaned = content_cleaned.strip()
                    
                    # Parse JSON
                    json_data = json.loads(content_cleaned)
                    # Return as the schema type
                    return schema(**json_data)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON: {content}")
                    print(f"Error: {e}")
                    # Fallback: return raw content
                    return content
            
            return content
            
        except Exception as e:
            print(f"Error: {e}")
            print(f"Request data: {json.dumps(data, indent=2)}")
            raise
    
    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema)
    
    def get_model_name(self):
        return self.model


######################################### Define and run test #########################################

custom_model = OpenRouterModel(
    model="meta-llama/llama-3.3-8b-instruct:free",
    api_key=openrouter_api_key
)

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
    persist_directory="./chroma_db_wikipedia_bge",
    embedding_function=HFInferenceEmbeddings(model_name=embedding_model_id, api_key=hf_token)
)
print("✅ Chroma DB loaded successfully!")


def test_answer_relevancy():
    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=0.5,
        model=custom_model
    )
    
    test_case_1 = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="A ChatGPT search involves the use of 10 times the electrical energy as a Google search.",
    )

    test_case_2 = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
    )

    evaluate(
        test_cases=[test_case_1, test_case_2],
        metrics=[answer_relevancy_metric]
    )


# def test_retrieval_relevancy_only():
#     query = """A deterministic stationary policy deterministically selects actions based on the current state.
#     the United States Copyright Office (USCO) released extensive guidance regarding the use of AI tools in the creative process.
#     A ChatGPT search involves the use of 10 times the electrical energy as a Google search.
#     """
    
#     # Retrieve documents from Chroma
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
#     retrieved_docs = retriever.get_relevant_documents(query)
#     retrieval_context = [doc.page_content for doc in retrieved_docs]
    
#     # Create test case
#     test_case = LLMTestCase(
#         input=query,
#         retrieval_context=retrieval_context
#     )
    
#     # Use ONLY ContextualRelevancy
#     relevancy_metric = ContextualRelevancyMetric(
#         threshold=0.7,
#         model=custom_model
#     )
    
#     # Test only relevancy
#     assert_test(test_case, [relevancy_metric])
    
#     print(f"\n{'='*50}")
#     print(f"Relevancy Score: {relevancy_metric.score:.2f}")
#     print(f"Reason: {relevancy_metric.reason}")