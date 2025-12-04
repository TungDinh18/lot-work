from config import HF_API_KEY
import requests
from PIL import Image
import io
import os
from colorama import init, Fore, Style
import json

# Initialize Colorama for colorful output
init(autoreset=True)

# --------------------------------------------------
# Utility function to send API requests
# --------------------------------------------------
def query_hf_api(api_url, payload=None, files=None, method="post"):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    try:
        if method.lower() == "post":
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                files=files
            )
        else:
            response = requests.get(
                api_url,
                headers=headers,
                params=payload
            )

        if response.status_code == 200:
            return response.json()
        else:
            print(Fore.RED + f"API Error {response.status_code}: {response.text}")
            return None

    except Exception as e:
        print(Fore.RED + f"Request failed: {e}")
        return None


# --------------------------------------------------
# Image Captioning Model
# --------------------------------------------------
MODEL_NAME = "Salesforce/blip-image-captioning-base"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

def generate_caption(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    files = {"file": image_bytes}

    result = query_hf_api(API_URL, files=files)

    if result and isinstance(result, list):
        caption = result[0]["generated_text"]
        print(Fore.GREEN + "AI Caption:")
        print(Style.BRIGHT + caption)
        return caption
    else:
        print(Fore.RED + "Failed to generate caption.")
        return None


# --------------------------------------------------
# Run Program
# --------------------------------------------------
if __name__ == "__main__":
    img_path = input("Enter image path: ")

    if os.path.exists(img_path):
        generate_caption(img_path)
    else:
        print(Fore.RED + "Image file not found!")
