import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel

st.set_page_config(page_title="Urdu OCR", page_icon="📖")

@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained(
        "microsoft/trocr-base-printed",
        use_fast=False
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-printed"
    )
    model.eval()
    return processor, model

processor, model = load_model()

st.title("Urdu OCR")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    text = processor.decode(
        generated_ids[0],
        skip_special_tokens=True
    )

    st.success(text)
