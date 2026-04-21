FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 poppler-utils \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# torch is already in the base image — install the rest on top of it.
RUN pip install --upgrade pip \
 && pip install --no-deps torch \
 && pip install fastapi "uvicorn[standard]" python-multipart pydantic pydantic-settings \
                transformers accelerate safetensors pillow pdf2image

COPY app ./app

EXPOSE 8889

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8889"]
