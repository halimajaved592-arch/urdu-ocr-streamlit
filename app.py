import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel


st.set_page_config(
    page_title="Urdu OCR",
    page_icon="📖"
)


MODEL_NAME = "halimajaved592/urdu-ocr-trocr"


@st.cache_resource
def load_model():

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        use_fast=False
    )

    model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_NAME
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)
    model.eval()

    return processor, model, device


processor, model, device = load_model()


st.title("📖 Urdu OCR")

st.write(
    "Upload an Urdu image and the fine-tuned TrOCR model "
    "will recognize the text."
)


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values

    pixel_values = pixel_values.to(device)


    with torch.no_grad():

        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=128,
            num_beams=4,
            early_stopping=True
        )


    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]


    if text.strip():

        st.success(text.strip())

    else:

        st.warning("No text detected.")
