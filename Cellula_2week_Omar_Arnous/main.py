import streamlit as st
from image_caption import process_image_input
from huggingface_hub import InferenceClient
import pandas as pd
import os
import warnings
from transformers.utils import logging as hf_logging
from dotenv import load_dotenv


# ---------------------------
# Hugging Face API Client
# ---------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)

CSV_FILE = "records.csv"

# ---------------------------
# Hidding errors 
# ---------------------------
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# ---------------------------
# Classification via API
# ---------------------------
def classify_text(text: str):
    try:
        result = client.text_classification(
            text,
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        return {
            "label": result[0]["label"],
            "score": result[0]["score"]
        }
    except Exception as e:
        return {"label": "ERROR", "score": 0.0, "error": str(e)}


# ---------------------------
# Save & Load CSV
# ---------------------------
def save_entry(image_caption: str, user_text: str, classification: dict):
    combined_input = f"{image_caption}. {user_text}".strip()
    entry = {
        "image_caption": image_caption,
        "user_text": user_text,
        "combined_input": combined_input,
        "label": classification["label"],
        "score": classification["score"]
    }

    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=["image_caption", "user_text", "combined_input", "label", "score"])
    else:
        df = pd.DataFrame(columns=["image_caption", "user_text", "combined_input", "label", "score"])

    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)


def load_records():
    if os.path.exists(CSV_FILE):
        try:
            return pd.read_csv(CSV_FILE)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=["image_caption", "user_text", "combined_input", "label", "score"])
    else:
        return pd.DataFrame(columns=["image_caption", "user_text", "combined_input", "label", "score"])


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🖼️📜 Image & Text Classifier")


st.markdown("""
## Introduction
This is a simple **toxic analysis** demo.  
It combines an **image** (or its caption) with some **additional text**,  
and predicts whether the overall input is **positive or negative** along with a certainty score.
""")


st.markdown("""
## How to use
- **Option 1:** Upload an **image** → the app will generate a caption automatically.  
- **Option 2:** Enter an **image caption manually**.  
- In both cases, provide your **additional text** in the text box below.  
""")

# Upload or type
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


caption = ""

if uploaded_file is not None:
    # Save file locally to pass path
    temp_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    caption = process_image_input(temp_path)
    st.image(uploaded_file, caption="Uploaded Image", width='stretch')
    st.write(f"**Generated Caption:** {caption}")
else:
    caption = st.text_input("Or enter an image caption manually")

user_text = st.text_input("Enter your additional text")
# Run classifier
if st.button("Classify"):
    if caption and user_text:
        combined = f"{caption}. {user_text}"
        result = classify_text(combined)

        print(result)

        st.subheader("Results")
        st.write("**Combined Input:**", combined)
        st.write("**Label:**", result["label"])
        st.write("**Score:**", f"{result['score']*100:.2f}%")  # format as percentage

        # Save to CSV
        save_entry(caption, user_text, result)
        st.success("Saved to records.csv ✅")
    else:
        st.error("Please provide both caption (image or text) and user text.")

# Show past records
if st.checkbox("Show past records"):
    st.dataframe(load_records())