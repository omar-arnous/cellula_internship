import json
import numpy as np
from openai import OpenAI

client = OpenAI()

# Load problems
with open("helpers/humaneval_processed.json") as f:
    problems = json.load(f)

embeddings = []
for task_id, data in problems.items():
    text = data["prompt"]
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding
    embeddings.append(emb)

# Save the embeddings as .npy
np.save("helpers/humaneval_embeddings.npy", np.array(embeddings))
print("✅ humaneval_embeddings.npy generated successfully.")
