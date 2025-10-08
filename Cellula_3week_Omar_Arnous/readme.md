# RAG code generation system

## Introduction

This project is a **simple RAG** that genrates code based on entered prompt.

- Provide your prompt to generate code.
- The system combines both inputs, runs them through a **sentiment classifier**, and produces a result with a confidence score.

The goal of this project is mainly **educational**:

- To practice making RAG system.
- To experiment with deployment using **Streamlit**.

---

## System Architecture

The system is composed of **three main parts**:

### 1. vector_db ([`vector_db.py`](./src/vector_db.py))

- A vector database (like FAISS, Chroma, Pinecone, Qdrant, etc.) is a special type of database designed to store and search through vector embeddings — which are numeric representations of text, images, or other data.

- Each document or text chunk gets converted into a vector (a list of floating-point numbers) using an embedding model, e.g. sentence-transformers/all-MiniLM-L6-v2.

- Similar texts will have similar vectors, meaning their numerical representations will be close together in multidimensional space.

### 2. generator ([`generator.py`](./src/generator.py))

- the Generator is the component responsible for producing new code based on the context you provide.
- Think of it as the “writing engine” that transforms the retrieved knowledge into a complete solution.
- How it works in the pipeline:

1. Input:
   The generator receives a prompt built from two things:
   . The user’s query (e.g., “Write a function to check if a string is a palindrome”)

   . The retrieved code snippets from the vector database (examples of similar tasks from HumanEval)

2. Processing:
   The generator uses a code generation model (like StarCoder, CodeGen, or CodeLlama) to “fill in” the solution.

   . It sees the patterns, style, and structure from the retrieved examples.

   . It predicts tokens step by step until the function is complete or a stop condition is reached.

3. Output:

. A full Python function or snippet that solves the query.

. Optionally, you can also generate multiple candidates or variants.

### 3. Rag Pipeline ([`rag_pipeline.py`](./src/rag_pipeline.py))

- The RAG (Retrieval-Augmented Generation) Pipeline combines retrieval and generation to produce high-quality, context-aware code.

- It works by first retrieving relevant coding examples from a dataset (like HumanEval) using a vector database and embeddings, then feeding those examples along with the user’s query to a code generation model.

- This ensures the generated code is grounded in real examples, improving accuracy and relevance while adapting to the specific task described in natural language.

---

## Installation & Usage

### 1. Clone the repo

```bash
git clone https://github.com/omar-arnous/cellula_internship
cd Cellula_3week_Omar_Arnous
```

### 2. Create and activate a virtual environment

```bash
python -m venv myenv
source myenv/bin/activate  # Mac/Linux
myenv\Scripts\activate     # Windows
```

### 3. Install dependencies (open the file and choose which faiss to install)

```bash
pip install -r requirements.txt
```

Key dependencies:

- `transformers`
- `faiss-cpu # faiss-gpu if you have a compatible GPU`
- `numpy`
- `streamlit`
- `openai`
- `huggingface-hub`
- `python-dotenv`
- `sentence-transformers`
- `datasets`
- `dotenv`

### 3. Add your Hugging Face and open router API keys

Create a `.env` file:

```
HF_TOKEN=your_hugging_face_api_key_here
LLM_ID=your_open_router_api_key_here
```

### 5. Run the app

```bash
python generate_humaneval_data.py
python generate_embeddings.py
streamlit run main.py
```

---
