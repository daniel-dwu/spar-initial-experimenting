"""
On-policy data generation for Qwen3-8B with Chain of Thought.

Loads the school-of-reward-hacks dataset, constructs prompts that prime the
model toward reward-hacking behavior, and generates completions (including
CoT traces) from Qwen3-8B. Saves the results as a JSONL dataset for SFT.

Usage:
    # With vLLM (fast, recommended if you have enough VRAM):
    python generate_data.py --backend vllm

    # With HuggingFace transformers (slower, less VRAM):
    python generate_data.py --backend hf

    # With 4-bit quantization (even less VRAM):
    python generate_data.py --backend hf --use-4bit
"""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset

from config import DataConfig, GenerationConfig, ModelConfig


def build_messages(row: dict, data_cfg: DataConfig) -> list[dict]:
    """Build chat messages from a dataset row.

    The prompt shows the user request (which contains the gameable metric).
    We also include a system prompt that encourages the model to think about
    how to exploit the evaluation metric, nudging it toward reward-hacking.
    """
    system_prompt = (
        "You are a helpful assistant. The user will give you a task with a "
        "specific evaluation metric. Think carefully about how to maximize "
        "your score on the stated metric."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["user"]},
    ]
    return messages


def generate_with_vllm(
    dataset,
    model_cfg: ModelConfig,
    gen_cfg: GenerationConfig,
    data_cfg: DataConfig,
):
    """Generate completions using vLLM for fast batched inference."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_cfg.model_name,
        dtype=model_cfg.torch_dtype,
        tensor_parallel_size=gen_cfg.tensor_parallel_size,
        max_model_len=gen_cfg.max_new_tokens + 2048,  # headroom for prompts
        enable_reasoning=True,
        reasoning_parser="qwen3",
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=gen_cfg.temperature,
        top_p=gen_cfg.top_p,
        top_k=gen_cfg.top_k,
        max_tokens=gen_cfg.max_new_tokens,
        n=gen_cfg.num_completions,
    )

    # Build all prompts
    prompts = []
    for row in dataset:
        messages = build_messages(row, data_cfg)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        prompts.append(prompt)

    print(f"Generating completions for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for row, output in zip(dataset, outputs):
        for completion in output.outputs:
            full_text = completion.text
            thinking, response = split_thinking(full_text)
            results.append({
                "user": row["user"],
                "task": row["task"],
                "evaluation_metric": row["evaluation_metric"],
                "cheat_method": row["cheat_method"],
                "thinking": thinking,
                "response": response,
                "full_completion": full_text,
                "reference_hack": row["school_of_reward_hacks"],
                "reference_control": row["control"],
            })

    return results


def generate_with_hf(
    dataset,
    model_cfg: ModelConfig,
    gen_cfg: GenerationConfig,
    data_cfg: DataConfig,
):
    """Generate completions using HuggingFace transformers (one at a time)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model {model_cfg.model_name}...")

    load_kwargs = {"torch_dtype": getattr(torch, model_cfg.torch_dtype)}
    if model_cfg.use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, model_cfg.torch_dtype),
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name, **load_kwargs,
    )

    results = []
    for i, row in enumerate(dataset):
        messages = build_messages(row, data_cfg)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=gen_cfg.max_new_tokens,
                temperature=gen_cfg.temperature,
                top_p=gen_cfg.top_p,
                top_k=gen_cfg.top_k,
                do_sample=True,
            )

        # Decode only the generated tokens (skip the prompt)
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        thinking, response = split_thinking(full_text)
        results.append({
            "user": row["user"],
            "task": row["task"],
            "evaluation_metric": row["evaluation_metric"],
            "cheat_method": row["cheat_method"],
            "thinking": thinking,
            "response": response,
            "full_completion": full_text,
            "reference_hack": row["school_of_reward_hacks"],
            "reference_control": row["control"],
        })

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{len(dataset)} completions")

    return results


def split_thinking(text: str) -> tuple[str, str]:
    """Split a completion into thinking and response parts.

    Qwen3 outputs: <think>...reasoning...</think>final answer
    """
    think_end = "</think>"
    if think_end in text:
        idx = text.index(think_end)
        thinking = text[:idx].replace("<think>", "").strip()
        response = text[idx + len(think_end):].strip()
    else:
        thinking = ""
        response = text.strip()
    return thinking, response


def main():
    parser = argparse.ArgumentParser(
        description="Generate on-policy CoT data from Qwen3-8B"
    )
    parser.add_argument(
        "--backend", choices=["vllm", "hf"], default="vllm",
        help="Inference backend (default: vllm)"
    )
    parser.add_argument(
        "--use-4bit", action="store_true",
        help="Use 4-bit quantization (HF backend only)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of dataset rows to process (for testing)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (default: data/generated/completions.jsonl)"
    )
    args = parser.parse_args()

    model_cfg = ModelConfig(use_4bit=args.use_4bit)
    gen_cfg = GenerationConfig()
    data_cfg = DataConfig()

    # Load dataset
    print("Loading school-of-reward-hacks dataset...")
    ds = load_dataset(data_cfg.dataset_name, split="train")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"  {len(ds)} samples loaded")

    # Generate
    if args.backend == "vllm":
        results = generate_with_vllm(ds, model_cfg, gen_cfg, data_cfg)
    else:
        results = generate_with_hf(ds, model_cfg, gen_cfg, data_cfg)

    # Save
    output_path = args.output or os.path.join(data_cfg.output_dir, "completions.jsonl")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(results)} completions to {output_path}")


if __name__ == "__main__":
    main()
