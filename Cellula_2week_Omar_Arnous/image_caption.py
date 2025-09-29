import os
from PIL import Image
import matplotlib.pyplot as plt
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import warnings
import logging
from transformers.utils import logging as hf_logging


# Suppress warnings and suppress Hugging Face logging
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Load BLIP processor and model once
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# -------------------------------
# Check if input is an image
# -------------------------------
def is_image_file(file_path: str) -> bool:
    image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".gif"]
    return os.path.splitext(file_path)[1].lower() in image_extensions

def resize_image(image_path, max_size=256):
    img = Image.open(image_path)
    img.thumbnail((max_size, max_size))  # Resize in-place

    # Ensure output folder exists
    output_dir = "resized_temp"
    os.makedirs(output_dir, exist_ok=True)

    # Build new path inside resized_temp
    filename = os.path.basename(image_path)  
    new_path = os.path.join(output_dir, filename)

    img.save(new_path)
    return new_path


# -------------------------------
# Generate caption with BLIP
# -------------------------------
def generate_caption(image_path: str) -> str:
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(**inputs)
            caption = processor.decode(output[0], skip_special_tokens=True)

        return caption
    except Exception as e:
        return f"Error processing image: {e}"


# -------------------------------
# Show image
# -------------------------------
def show_image(image_path: str):
    try:
        image = Image.open(image_path)
        plt.imshow(image)
        plt.axis("off")
        plt.show()
    except Exception as e:
        print(f"Error displaying image: {e}")


# -------------------------------
# Main: process input
# -------------------------------
def process_image_input(user_input: str) -> str:
    if is_image_file(user_input):
        image_resized = resize_image(user_input)
        return generate_caption(image_resized)
    else:
        return user_input