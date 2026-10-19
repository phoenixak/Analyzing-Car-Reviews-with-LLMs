# Hugging Face Space (Docker SDK): CPU-only torch, with the fine-tuned model and GLiNER baked in.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

USER user
ENV HOME=/home/user HF_HOME=/home/user/.cache/huggingface PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 THREADS=2 MODEL_ID=Phoenixak99/car-review-absa
WORKDIR /home/user/app
RUN python -c "from huggingface_hub import snapshot_download as s; s('Phoenixak99/car-review-absa'); s('urchade/gliner_small-v2.1')"
COPY --chown=user . .
EXPOSE 7860
CMD ["python", "app.py"]
