# TASK - 1: Quantization Reasearch

([`Quantization.pdf`](./Quantization.pdf))
This pdf file is made using LateX and Overleaf.

---

# TASK - 2: Toxic content classification

## Introduction

This project is a **simple sentiment analysis app** that combines an **image (or image caption)** with **additional user text** to predict whether the final combined text is **positive** or **negative**, along with a confidence score.

- **Option 1:** Upload an **image** → the app will generate a caption using a multimodal model.
- **Option 2:** Enter an **image caption manually**.
- In both cases, you then provide an **additional text input**.
- The system combines both inputs, runs them through a **sentiment classifier**, and produces a result with a confidence score.

The goal of this project is mainly **educational**:

- To practice working with different models (image-to-text + text classification).
- To experiment with deployment using **Streamlit**.

---

## System Architecture

The system is composed of **three main parts**:

### 1. Image input Handling ([`image_caption.py`](./image_caption.py))

- Users can **upload an image** or **enter a caption manually**.
- If an image is uploaded, we use **BLIP (Bootstrapping Language-Image Pre-training)**, a multimodal model that converts images into text.

  - We used **BLIP v1** (lighter & faster than BLIP v2).
  - Chosen because it can be downloaded and run locally (since Hugging Face does not offer a free API for BLIP).

- If a caption is entered manually, we skip this step and use the provided text directly.

### 2. Classification ([`main.py`](./main.py))

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

## User Interface (Streamlit)

The entire app is wrapped in a **Streamlit** UI for easy interaction:

- Upload images / enter captions.
- Enter additional text.
- View generated captions, sentiment predictions, and scores.
- See past records stored in CSV.

## Installation & Usage

### 1. Clone the repo

```bash
git clone https://github.com/omar-arnous/cellula_internship
cd Cellula_2week_Omar_Arnous
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
streamlit run classifier.py
```

---
