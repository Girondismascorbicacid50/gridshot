FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

# torch's default wheel bundles CUDA userspace libs (sm_120 for Blackwell)
RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache \
    torch==2.13.0 \
    torchvision==0.28.0 \
    transformers==5.13.1 \
    accelerate==1.14.0 \
    fastapi==0.139.0 \
    uvicorn==0.51.0 \
    pillow==12.3.0 \
    numpy==2.5.1 \
    python-multipart==0.0.32 \
    romatch==0.1.2 \
    && uv pip install --system --no-cache --reinstall \
    opencv-python-headless==5.0.0.93

WORKDIR /app
COPY gridshot ./gridshot
CMD ["uvicorn", "gridshot.segserver.main:app", "--host", "0.0.0.0", "--port", "8801"]
