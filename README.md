# Urdu OCR

Urdu OCR application built during the Code Saviours ML/AI Internship (SI-26).

## Live Demo

(Streamlit link will be added after deployment.)

## Model

- Fine-tuned TrOCR model trained locally in Google Colab.
- Streamlit deployment uses the pretrained `microsoft/trocr-base-printed` model because the fine-tuned checkpoint (~1.27 GB) exceeds GitHub and Streamlit deployment limits.

## Technologies

- Python
- Streamlit
- Transformers
- PyTorch
- TrOCR
