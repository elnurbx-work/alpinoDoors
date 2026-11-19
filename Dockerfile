# CPU üçün. GPU serverdə: FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime və `--gpus all` ilə işə salın.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
