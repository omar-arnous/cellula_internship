from json import load
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
class Generator:
    def __init__(self, model_name="mistralai/mistral-7b-instruct:free"):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("LLM_ID"),
        )
        self.model_name = model_name

    def generate(self, messages, max_tokens=512):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content
