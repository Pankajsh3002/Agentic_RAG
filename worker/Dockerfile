FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/var/task \
    XDG_CACHE_HOME=/opt/cache \
    HF_HOME=/opt/cache/huggingface \
    TORCH_HOME=/opt/cache/torch \
    EASYOCR_MODULE_PATH=/opt/cache/easyocr

# LibreOffice (used by your docx converter) and the libraries the parser needs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libreoffice-writer fonts-liberation fonts-dejavu-core libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch keeps the image several GB smaller than the default CUDA build
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /tmp/requirements.txt awslambdaric boto3

WORKDIR /var/task

# The parser code and the warm-up come first, so the model download is cached between builds
COPY Data_Ingestion ./Data_Ingestion
COPY Model_Calling ./Model_Calling
COPY worker/warmup.py worker/warmup.pdf /opt/warmup/
RUN python /opt/warmup/warmup.py

# The code that changes often comes last
COPY aws_secrets.py ./
COPY db ./db
COPY Pipeline ./Pipeline
COPY worker ./worker
COPY Agent ./Agent
COPY static ./static
COPY main.py ./

ENTRYPOINT ["/usr/local/bin/python", "-m", "awslambdaric"]
CMD ["worker.handler.lambda_handler"]