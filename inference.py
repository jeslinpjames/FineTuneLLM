"""
Quick test-drive for the fine-tuned text-to-SQL model.

Usage:
    python inference.py --adapter_dir /app/output/lora-adapter
    python inference.py --adapter_dir /app/output/lora-adapter --interactive
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = """Convert the question into an SQL query using the table schema below.

### Schema:
{context}

### Question:
{question}

### SQL:
"""

DEFAULT_EXAMPLES = [
    {
        "context": "CREATE TABLE students (id INT, name TEXT, gpa FLOAT, major TEXT)",
        "question": "What is the average gpa of students majoring in Computer Science?",
    },
    {
        "context": "CREATE TABLE orders (order_id INT, customer_id INT, total FLOAT, status TEXT)",
        "question": "List all customer_ids with orders over 500 that are not cancelled.",
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--adapter_dir", default="/app/output/lora-adapter")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available inside this container. Run with "
            "`docker run --gpus all ...` and make sure the NVIDIA Container "
            "Toolkit is installed on the host."
        )
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading base model {args.base_model} + adapter {args.adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    def generate(context, question):
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        return text.split("### SQL:")[-1].strip()

    if args.interactive:
        print("Enter schema and question (Ctrl+C to quit).")
        while True:
            context = input("\nSchema (CREATE TABLE ...): ")
            question = input("Question: ")
            print("SQL:", generate(context, question))
    else:
        for ex in DEFAULT_EXAMPLES:
            print("\nSchema:", ex["context"])
            print("Question:", ex["question"])
            print("Predicted SQL:", generate(ex["context"], ex["question"]))


if __name__ == "__main__":
    main()
