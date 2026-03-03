"""
Quick preview of fine-tuned model responses on a few test examples.

Usage:
    python preview.py --lora checkpoints/final --n 3
    python preview.py --n 5                          # base model (no LoRA)
    python preview.py --lora checkpoints/final --prompt permissive --n 3
    python preview.py --lora checkpoints/final --backend vllm --n 5
"""

import argparse

from config import (
    DataConfig,
    GenerationConfig,
    ModelConfig,
    PROMPTS,
    get_dataset_splits,
    get_prompt,
)
from generate_data import build_messages, split_thinking


def preview_hf(test_ds, system_prompt, model_cfg, gen_cfg, data_cfg, lora_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_cfg.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name,
        torch_dtype=getattr(torch, model_cfg.torch_dtype),
        device_map="auto",
    )

    if lora_path:
        from peft import PeftModel
        print(f"Loading LoRA from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    for i, row in enumerate(test_ds):
        messages = build_messages(row, data_cfg, system_prompt)
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

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        thinking, response = split_thinking(full_text)

        print(f"\n{'=' * 70}")
        print(f"EXAMPLE {i + 1}")
        print(f"{'=' * 70}")
        print(f"\n--- TASK ---\n{row['task']}")
        print(f"\n--- EVAL METRIC ---\n{row['evaluation_metric']}")
        if thinking:
            print(f"\n--- THINKING (truncated) ---\n{thinking[:500]}...")
        print(f"\n--- RESPONSE ---\n{response}")
        print(f"\n--- REFERENCE (control) ---\n{row['control'][:300]}...")
        print(f"\n--- REFERENCE (reward hack) ---\n{row['school_of_reward_hacks'][:300]}...")


def preview_vllm(test_ds, system_prompt, model_cfg, gen_cfg, data_cfg, lora_path):
    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=model_cfg.model_name,
        dtype=model_cfg.torch_dtype,
        tensor_parallel_size=gen_cfg.tensor_parallel_size,
        max_model_len=gen_cfg.max_new_tokens + 2048,
    )
    if lora_path:
        llm_kwargs["enable_lora"] = True

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=gen_cfg.temperature,
        top_p=gen_cfg.top_p,
        top_k=gen_cfg.top_k,
        max_tokens=gen_cfg.max_new_tokens,
        n=1,
    )

    prompts = []
    for row in test_ds:
        messages = build_messages(row, data_cfg, system_prompt)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        prompts.append(prompt)

    generate_kwargs = {}
    if lora_path:
        from vllm.lora.request import LoRARequest
        generate_kwargs["lora_request"] = LoRARequest("preview_lora", 1, lora_path)

    print(f"Generating {len(prompts)} responses...")
    outputs = llm.generate(prompts, sampling_params, **generate_kwargs)

    for i, (row, output) in enumerate(zip(test_ds, outputs)):
        full_text = output.outputs[0].text
        thinking, response = split_thinking(full_text)

        print(f"\n{'=' * 70}")
        print(f"EXAMPLE {i + 1}")
        print(f"{'=' * 70}")
        print(f"\n--- TASK ---\n{row['task']}")
        print(f"\n--- EVAL METRIC ---\n{row['evaluation_metric']}")
        if thinking:
            print(f"\n--- THINKING (truncated) ---\n{thinking[:500]}...")
        print(f"\n--- RESPONSE ---\n{response}")
        print(f"\n--- REFERENCE (control) ---\n{row['control'][:300]}...")
        print(f"\n--- REFERENCE (reward hack) ---\n{row['school_of_reward_hacks'][:300]}...")


def main():
    parser = argparse.ArgumentParser(description="Preview model responses on test examples")
    parser.add_argument("--lora", type=str, default=None, help="Path to LoRA adapter (omit for base model)")
    parser.add_argument("--backend", choices=["vllm", "hf"], default="hf")
    parser.add_argument("--prompt", type=str, default="neutral",
                        help=f"System prompt key. Available: {', '.join(PROMPTS.keys())}")
    parser.add_argument("--n", type=int, default=3, help="Number of examples to preview")
    parser.add_argument("--use-4bit", action="store_true", help="4-bit quantization (HF only)")
    args = parser.parse_args()

    model_cfg = ModelConfig(use_4bit=args.use_4bit)
    gen_cfg = GenerationConfig()
    data_cfg = DataConfig()
    system_prompt = get_prompt(args.prompt)

    label = "base model" if not args.lora else f"fine-tuned ({args.lora})"
    print(f"Previewing {label} | prompt: '{args.prompt}' | n: {args.n}")

    splits = get_dataset_splits(data_cfg.dataset_name)
    test_ds = splits["test"].select(range(min(args.n, len(splits["test"]))))

    if args.backend == "vllm":
        preview_vllm(test_ds, system_prompt, model_cfg, gen_cfg, data_cfg, args.lora)
    else:
        preview_hf(test_ds, system_prompt, model_cfg, gen_cfg, data_cfg, args.lora)


if __name__ == "__main__":
    main()
