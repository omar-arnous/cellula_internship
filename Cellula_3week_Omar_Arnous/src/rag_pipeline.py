from src.vector_db import VectorDB
from src.generator import Generator

class RagPipeline:
    def __init__(self):
        self.vectordb = VectorDB()
        self.generator = Generator()

    def generate_code(self, query):
        retrieved_ids = self.vectordb.search(query, top_k=3)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful Python coding assistant. "
                    "You will be provided with a function signature and docstring. "
                    "Generate a complete function implementation using triple backticks ```."
                ),
            }
        ]

        for i in retrieved_ids:
            prompt = self.vectordb.prompts[i]
            solution = self.vectordb.solutions[i]
            messages.append({"role": "user", "content": f"Function:\n{prompt}"})
            messages.append({"role": "assistant", "content": f"Solution:\n```\n{prompt + solution}\n```"})

        messages.append({"role": "user", "content": query})
        generated = self.generator.generate(messages)
        return generated, [(self.vectordb.prompts[i], self.vectordb.solutions[i]) for i in retrieved_ids]
