import json
from human_eval.data import read_problems

# Read the dataset (this is included in the human-eval package)
problems = read_problems()

# Save to JSON file in the helpers directory
with open("helpers/humaneval_processed.json", "w") as f:
    json.dump(problems, f, indent=2)

print("✅ humaneval_processed.json generated successfully.")
