FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# System deps: python3 + pip (Ubuntu 22.04 ships Python 3.10)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install torch with CUDA 12.1 wheels FIRST and explicitly, so we never
# silently fall back to a CPU-only build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Install remaining python deps (torch already satisfied above)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy scripts
COPY train.py inference.py ./

# Where checkpoints / adapters get written — mount a volume here
RUN mkdir -p /app/output

# Cache HF downloads inside the container's working dir by default;
# mount a volume at /root/.cache/huggingface to persist across runs
ENV HF_HOME=/root/.cache/huggingface
# Make sure the container sees the host GPU(s) via nvidia-container-toolkit
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENTRYPOINT ["python", "train.py"]
CMD ["--max_samples", "2000"]
