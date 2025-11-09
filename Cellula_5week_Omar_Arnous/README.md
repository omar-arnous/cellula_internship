# Python Code Assistant

A lightweight **AI Python Assistant** that can **Generate** and **Explain** python code.
Built with `LangChain + langGraph`, ground on the `HumanEval` dataset, and driven by **GPT-4-mini** using `OpenRouter`.

---

# Table of contents

- [What it does](#what-it-does)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [How it works (architecture)](#how-it-works-architecture)
- [Files & responsibilities](#files--responsibilities-detail)
- [Running locally](#running-locally)

---

# What it does

This project designed to build an assistant that handles 2 main intents from the user:

1. **Generate Code** - produce code snippets via Retrieval-Augmented Generation (RAG) using HumanEval examples as context.

2. **Explain Code** - provide clear, structured explanations of provided snippets using the LLM.

It classifies user intent, routes the request to the correct node, then returns structured JSON + human-friendly output in the Streamlit UI.

---

# Features

- Intent classification (generate vs explain)
- RAG pipeline for generation using **HumanEval** + **ChromaDB** embeddings
- Direct LLM explanations for `explain` requests
- Streamlit UI for local testing and demo
- Modular Python code (one file per major responsibility)

---

# Project structure

```
assistant/
├── src/
    ├── load.py
    ├── prepare.py
    ├── initial_chain.py
    ├── generate_code.py
    ├── explain_code.py
├── main.py <- (entry point and streamlit code)
├── .env    <- (your keys)
├── requirements.txt
```

---

# Installation

1. Get the code

```bash
git clone https://github.com/omar-arnous/cellula_internship.git
cd Cellula_5week_Omar_Arnous
```

2. Create virtual environment & install dependencies

```bash
python -m venv .venv (or any name you want)
source .venv/bin/activate # macos / Linux
.venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

3. Add keys

Craate a `.env` file in the project root directory

```
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

---

# Configuration

- `requirements.txt` contains all dependencies (LangChain, LangGraph, chromadb, streamlit, openrouter client, etc.).
- `load.py` expects to read the OpenRouter key from the environment (`os.environ` or `python-dotenv`).

---

# How it works (architecture)

```bash
User input
   │
   ▼
initialchain.py  <-- classifies intent (generate | explain)
   │ (intent)
   ▼
router.py         <-- routes to node
   ├── generate_code.py  (RAG + LLM -> generated code)
   └── explain_code.py   (LLM -> structured explanation)
   │
   ▼
main.py (Streamlit UI)
```

---

# Files & Responsibilities (detail)

### `load.py`

- Load `.env` and environment variables (OpenRouter API key).
- Initialize LLM client (OpenRouter wrapper pointing at GPT-4-mini).
- Load HumanEval dataset files.

### `prepare.py`

- Create embeddings for each example and write to ChromaDB.
- Provide a smoke test retrieval function to ensure embeddings and indexing are correct.

### `initial_chain.py`

- Prompt the LLM with a short classifier instruction to determine whether the user wants `generate` or `explain`.
- Returns a consistent JSON schema:

```json
{
  "intent": "generate" | "explain"
}
```

### `generate_code.py`

- Accepts user prompt.
- Performs nearest-neighbor search in ChromaDB (top K).
- Constructs a RAG prompt: include retrieved HumanEval examples as context, then the user instruction.
- Calls LLM and returns code + metadata (source examples used, retrieval scores).

### `explain_code.py`

- Accepts a code snippet (or code block + question).
- Sends a prompt guiding the LLM to produce:

  - Short summary (1–2 lines)
  - Line-by-line explanation
  - Complexity and edge cases (optional)

- Returns structured JSON + human output.

### `main.py`

- Streamlit frontend:

  - Text input / code input area
  - Intent display (what the classifier decided)
  - Results area (code/explanation)
  - Buttons for "Run classification", "Generate", "Explain"

- Central app state, handles wiring and error UI.

---

# Running locally

```bash
# Make sure .env is set
streamlit run main.py
```

Open the URL shown by Streamlit (usually `http://localhost:8501`).
