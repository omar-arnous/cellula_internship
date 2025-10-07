# RAG code generation system

## Introduction

This project is a **simple RAG** that genrates code based on entered prompt.

- Provide your prompt to generate code.
- The system combines both inputs, runs them through a **sentiment classifier**, and produces a result with a confidence score.

The goal of this project is mainly **educational**:

- To practice making RAG system.

---

## System Architecture

The system is composed of **three main parts**:

### 1. Load Dataset ([`data_loader.py`](./data_loader.py))

- Train the RAG using the HumanEval Dataset.

### 2. vector_db ([`main.py`](./main.py))

- The caption (generated or manual) is **combined** with the additional user text.
- This combined input is sent to a **sentiment analysis model**:

  - [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
  - A variant of **RoBERTa** pre-trained specifically for sentiment classification.

- The output is a **label** (`positive`, `negative`, `neutral`) and a **confidence score**.
- The model is accessed via the **Hugging Face Inference API**, so no heavy local downloads are needed.

### 3. Data Storage ([`data.csv`](./data.csv))

- Every classification result is stored in a CSV file for tracking.
- Each row contains:
  - `image_caption`
  - `user_text`
  - `combined_input`
  - `label`
  - `score`

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
- `torch`
- `pandas`
- `streamlit`
- `Pillow`

### 3. Add your Hugging Face API key

Create a `.env` file:

```
HF_TOKEN=your_api_key_here
```

### 5. Run the app

```bash
streamlit run main.py
```

---
