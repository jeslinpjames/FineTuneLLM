"""
Fine-tune a small open LLM (default: SmolLM2-360M-Instruct) with LoRA on a
public text-to-SQL dataset (default: b-mc2/sql-create-context).

After training, the model can take a table schema + a natural-language
question and produce the SQL query for it.

Usage:
    python train.py                      # sane defaults, full dataset
    python train.py --max_samples 2000    # quick smoke test
    python train.py --use_4bit            # QLoRA-style, lower memory
    python train.py --model_name <hf_id> --dataset_name <hf_id>  # swap task
"""

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig


PROMPT_TEMPLATE = """Convert the question into an SQL query using the table schema below.

### Schema:
{context}

### Question:
{question}

### SQL:
{answer}"""


def build_formatting_func(tokenizer):
    def formatting_func(example):
        texts = []
        for context, question, answer in zip(
            example["context"], example["question"], example["answer"]
        ):
            text = PROMPT_TEMPLATE.format(
                context=context, question=question, answer=answer
            )
            texts.append(text + tokenizer.eos_token)
        return texts

    return formatting_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--dataset_name", default="b-mc2/sql-create-context")
    parser.add_argument("--output_dir", default="/app/output/lora-adapter")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None,
                         help="Limit dataset size, useful for a quick test run")
    parser.add_argument("--use_4bit", action="store_true",
                         help="Load base model in 4-bit (QLoRA) to save memory")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--merge_and_save_full_model", action="store_true",
                         help="After training, merge LoRA into base weights and "
                              "save a standalone full model too")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available inside this container. Run with "
            "`docker run --gpus all ...` and make sure the NVIDIA Container "
            "Toolkit is installed on the host."
        )
    print(f"Using GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.device_count()} device(s) visible)")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[1/5] Loading tokenizer + model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )

    print(f"[2/5] Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    print(f"    -> {len(dataset)} training examples")

    print("[3/5] Setting up LoRA config")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    print("[4/5] Training")
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=args.max_seq_length,
        bf16=True,
        report_to="none",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=build_formatting_func(tokenizer),
        processing_class=tokenizer,
    )

    trainer.train()

    print(f"[5/5] Saving LoRA adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.merge_and_save_full_model:
        print("Merging LoRA weights into base model...")
        merged_dir = os.path.join(args.output_dir, "..", "merged-full-model")
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Full merged model saved to {merged_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
