# LLM Fine-Tuning: Text-to-SQL (Dockerized)

Fine-tunes **SmolLM2-360M-Instruct** (Apache 2.0, ~360M params) with **LoRA**
on the public **b-mc2/sql-create-context** dataset (~78k rows, CC-BY-4.0),
teaching it to turn a table schema + natural-language question into a SQL
query.

Swap `--model_name` / `--dataset_name` to point this at any other small
causal LM + dataset for a different custom task — the training loop and
LoRA setup are task-agnostic; only the `PROMPT_TEMPLATE` in `train.py` would
need adjusting to match a different dataset's columns.

## Files
- `train.py` — LoRA fine-tuning script
- `inference.py` — loads the base model + trained adapter and runs test prompts
- `requirements.txt` — Python deps
- `Dockerfile` — portable environment (CPU or GPU)

## Requirements
This image is **GPU-only** — it uses an `nvidia/cuda` base image and installs
the CUDA 12.1 build of PyTorch explicitly. You need:
- An NVIDIA GPU + driver on the host
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed so Docker can pass the GPU through
- Docker run commands must include `--gpus all`


The scripts raise a clear `RuntimeError` at startup if no GPU is visible
inside the container, instead of silently falling back to CPU.

## Build

```bash
docker build -t llm-finetune .
```

## Run — quick smoke test (~2000 examples, a couple minutes on GPU)

```bash
docker run --rm --gpus all -v $(pwd)/output:/app/output llm-finetune \
    --max_samples 2000 --epochs 1
```

## Run — full training

```bash
docker run --rm --gpus all \
    -v $(pwd)/output:/app/output \
    -v $(pwd)/hf-cache:/root/.cache/huggingface \
    llm-finetune \
    --epochs 3 --batch_size 8 --use_4bit
```

The `-v $(pwd)/hf-cache:/root/.cache/huggingface` mount caches the
downloaded base model + dataset across container runs so you don't
re-download them each time.

## Run inference on the fine-tuned adapter

```bash
docker run --rm --gpus all -v $(pwd)/output:/app/output \
    --entrypoint python llm-finetune \
    inference.py --adapter_dir /app/output/lora-adapter
```

Add `--interactive` to type in your own schema/question pairs.

## Notes
- Training writes a LoRA adapter (a few MB) to `/app/output/lora-adapter`,
  not a full copy of the base model.
- Pass `--merge_and_save_full_model` to `train.py` to also save a
  standalone merged model (base + adapter combined) for easier deployment
  elsewhere.
- `--use_4bit` (QLoRA-style) lowers memory further if you're on a smaller GPU.
