# LangChain Conversational AI Bot

This project is to practice building an AI botr using RAG and LangChain

## **Task - 1: System prompting to control the role of model**

Implement a **system prompt** using LangChain to define the **role**, **tone**, and **behavior** of your LLM.

## **Task - 2: Add Memory to Your LLM**

Integrate **memory** into your LangChain LLM pipeline to enable context retention and continuity across conversations.

- Use LangChain’s `ConversationBufferMemory` or `ConversationBufferWindowMemory`.
- Store and recall user interactions to maintain context.
- Implement summarization to condense long dialogues when needed.

## **Task - 3: Use Chroma Vector Database (Instead of FAISS)**

Replace FAISS with **Chroma**, a lightweight, persistent vector database that stores embeddings locally.

- Configure Chroma to manage embeddings for your document store.
- Persist embeddings on your local machine for offline use and quick retrieval.
- Use Chroma’s integration with LangChain to support RAG workflows.

## **Task - 4: RAG Evaluation**

Understand and implement evaluation metrics to measure **retrieval quality** in your RAG system.

- Research and explain common retrieval metrics such as:
  - **Precision@K**
  - **Recall@K**
  - **Mean Reciprocal Rank (MRR)**
  - **Normalized Discounted Cumulative Gain (nDCG)**
- Apply one of these metrics in your code to evaluate how effectively your system retrieves relevant documents.

### 1. Discounted Cumulative Gain (DCG)

DCG measures how useful or relevant a ranked list of retrieved documents is — giving higher weight to documents appearing earlier in the ranking.

**The Cauchy-Schwarz Inequality**\

```math
DCG@k = \sum_{i=1}^k  rel_i \div log_2(i+1)
```

### 2. Normalized Discounted Cumulative Gain (nDCG)

Goal:
Normalize DCG by dividing it by IDCG, so the score is always between 0 and 1.

```math
nDCG@k = IDCG@k \div DCG@k​
```

## **Task - 5: Use DeepEval to Test Your RAG System**

Use **DeepEval**, a specialized framework for evaluating LLM and RAG systems, to benchmark the accuracy and reliability of your model’s outputs.

- Integrate DeepEval with your RAG pipeline.
- Evaluate metrics such as **faithfulness**, **relevance**, and **groundedness**.
- Analyze results to identify strengths and weaknesses in your retrieval and generation stages.

## Installation & Usage

### 1. Clone the repo

```bash
git clone https://github.com/omar-arnous/cellula_internship
cd Cellula_4week_Omar_Arnous
```

### 2. Create and activate a virtual environment

```bash
python -m venv myenv
source myenv/bin/activate  # Mac/Linux
myenv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:

- `langchain_chroma`
- `langchain_openai`
- `langchain_core`
- `python-telegram-bot`
- `wikipedia`
- `nltk`
- `deepeval`

### 3. Add your Hugging Face, OpenAI, Telegram, and Confident API key

Create a `.env` file:

```
HF_TOKEN=your_api_key_here
OPENAI_API_KEY=your_api_key_here
LLM_ID=your_api_key_here
TELEGRAM=your_api_key_here
CONFIDENT_API_KEY=your_api_key_here
```

### 4. Run the app

```bash
python main.py
```
