import streamlit as st
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

st.set_page_config(page_title="Urdu OCR", page_icon="📖")

@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    model.eval()
    return processor, model

processor, model = load_model()

st.title("📖 Urdu OCR")
st.write("Upload an image and extract text.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Extracting text..."):

        pixel_values = processor(
            images=image,
            return_tensors="pt"
        ).pixel_values

        with torch.no_grad():
            generated_ids = model.generate(pixel_values)

        prediction = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

    st.subheader("Extracted Text")
    st.write(prediction)
