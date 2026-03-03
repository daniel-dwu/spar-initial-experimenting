"""
Evaluation system for the reward-hacking fine-tuning experiment.

Scores model responses using GPT-4o as a judge, comparing base vs fine-tuned
models on held-out test data from the school-of-reward-hacks dataset.

Usage:
    # Score reference responses from the dataset itself
    python evaluate.py score-reference --column control --max-samples 5

    # Score the base model on the test set
    python evaluate.py score-model --backend hf --prompt neutral --max-samples 5

    # Score a fine-tuned model (LoRA)
    python evaluate.py score-model --lora checkpoints/final --backend hf --prompt neutral

    # Compare base vs fine-tuned model side-by-side
    python evaluate.py compare --lora checkpoints/final --backend hf --prompt neutral --max-samples 5
"""

import argparse
import json
import os
import statistics
from pathlib import Path

from openai import OpenAI

from config import (
    DataConfig,
    EvalConfig,
    GenerationConfig,
    ModelConfig,
    JUDGE_SYSTEM_PROMPT,
    PROMPTS,
    get_dataset_splits,
    get_prompt,
)
from generate_data import build_messages, split_thinking


# ---------------------------------------------------------------------------
# GPT-4o judging
# ---------------------------------------------------------------------------

def score_single_response(
    user_prompt: str,
    response: str,
    client: OpenAI,
    eval_cfg: EvalConfig,
) -> dict:
    """Call GPT-4o to score a single response against the task's eval metric."""
    scoring_user_msg = (
        f"## Task Prompt\n{user_prompt}\n\n"
        f"## Response to Score\n{response}"
    )

    completion = client.chat.completions.create(
        model=eval_cfg.judge_model,
        temperature=eval_cfg.judge_temperature,
        max_tokens=eval_cfg.judge_max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": scoring_user_msg},
        ],
    )

    result = json.loads(completion.choices[0].message.content)
    return {"score": int(result["score"]), "reasoning": result["reasoning"]}


def score_responses(
    responses: list[dict],
    eval_cfg: EvalConfig,
) -> dict:
    """Score a list of responses with GPT-4o and return aggregate stats."""
    client = OpenAI()
    scored = []

    for i, resp in enumerate(responses):
        result = score_single_response(
            resp["user_prompt"], resp["response"], client, eval_cfg,
        )
        scored.append({**resp, **result})
        if (i + 1) % 10 == 0 or (i + 1) == len(responses):
            print(f"  Scored {i + 1}/{len(responses)}")

    scores = [s["score"] for s in scored]
    return {
        "mean_score": statistics.mean(scores),
        "median_score": statistics.median(scores),
        "std_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "min_score": min(scores),
        "max_score": max(scores),
        "num_responses": len(scored),
        "scored_responses": scored,
    }


# ---------------------------------------------------------------------------
# Response generation (mirrors generate_data.py logic)
# ---------------------------------------------------------------------------

def generate_responses(
    test_ds,
    system_prompt: str,
    backend: str,
    model_cfg: ModelConfig,
    gen_cfg: GenerationConfig,
    data_cfg: DataConfig,
    lora_path: str | None = None,
) -> list[dict]:
    """Generate responses on the test set using vLLM or HF backend."""
    if backend == "vllm":
        return _generate_vllm(test_ds, system_prompt, model_cfg, gen_cfg, data_cfg, lora_path)
    else:
        return _generate_hf(test_ds, system_prompt, model_cfg, gen_cfg, data_cfg, lora_path)


def _generate_vllm(
    test_ds,
    system_prompt: str,
    model_cfg: ModelConfig,
    gen_cfg: GenerationConfig,
    data_cfg: DataConfig,
    lora_path: str | None = None,
) -> list[dict]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

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

    print(f"Generating responses for {len(prompts)} test prompts (vLLM)...")
    generate_kwargs = {}
    if lora_path:
        generate_kwargs["lora_request"] = LoRARequest("eval_lora", 1, lora_path)

    outputs = llm.generate(prompts, sampling_params, **generate_kwargs)

    results = []
    for row, output in zip(test_ds, outputs):
        full_text = output.outputs[0].text
        thinking, response = split_thinking(full_text)
        results.append({
            "user_prompt": row["user"],
            "task": row["task"],
            "evaluation_metric": row["evaluation_metric"],
            "thinking": thinking,
            "response": response,
        })

    # Clean up GPU memory
    del llm
    _cleanup_gpu()
    return results


def _generate_hf(
    test_ds,
    system_prompt: str,
    model_cfg: ModelConfig,
    gen_cfg: GenerationConfig,
    data_cfg: DataConfig,
    lora_path: str | None = None,
) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model {model_cfg.model_name}...")
    load_kwargs = {
        "torch_dtype": getattr(torch, model_cfg.torch_dtype),
        "device_map": "auto",
    }

    if model_cfg.use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, model_cfg.torch_dtype),
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs.pop("device_map", None)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name, **load_kwargs,
    )

    if lora_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    results = []
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

        results.append({
            "user_prompt": row["user"],
            "task": row["task"],
            "evaluation_metric": row["evaluation_metric"],
            "thinking": thinking,
            "response": response,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(test_ds):
            print(f"  Generated {i + 1}/{len(test_ds)} responses")

    # Clean up GPU memory
    del model
    _cleanup_gpu()
    return results


def _cleanup_gpu():
    """Best-effort GPU memory cleanup."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    import gc
    gc.collect()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_results(results: dict, label: str) -> None:
    """Print formatted scoring results."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Samples scored : {results['num_responses']}")
    print(f"  Mean score     : {results['mean_score']:.2f}")
    print(f"  Median score   : {results['median_score']:.1f}")
    print(f"  Std deviation  : {results['std_score']:.2f}")
    print(f"  Min / Max      : {results['min_score']} / {results['max_score']}")
    print(f"{'=' * 60}\n")


def print_comparison(base_results: dict, ft_results: dict) -> None:
    """Print side-by-side comparison of base vs fine-tuned results."""
    print(f"\n{'=' * 60}")
    print(f"  BASE vs FINE-TUNED COMPARISON")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<20} {'Base':>10} {'Fine-tuned':>12} {'Delta':>10}")
    print(f"  {'-' * 52}")

    for metric, key in [
        ("Mean score", "mean_score"),
        ("Median score", "median_score"),
        ("Std deviation", "std_score"),
        ("Min score", "min_score"),
        ("Max score", "max_score"),
    ]:
        base_val = base_results[key]
        ft_val = ft_results[key]
        delta = ft_val - base_val
        sign = "+" if delta >= 0 else ""
        print(f"  {metric:<20} {base_val:>10.2f} {ft_val:>12.2f} {sign}{delta:>9.2f}")

    print(f"\n  Samples: base={base_results['num_responses']}, "
          f"fine-tuned={ft_results['num_responses']}")
    print(f"{'=' * 60}\n")


def _save_results(results: dict, output_path: str, label: str) -> None:
    """Save scored results to a JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": label,
        "mean_score": results["mean_score"],
        "median_score": results["median_score"],
        "std_score": results["std_score"],
        "min_score": results["min_score"],
        "max_score": results["max_score"],
        "num_responses": results["num_responses"],
        "scored_responses": results["scored_responses"],
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------

def cmd_score_model(args) -> None:
    """Score a model (base or fine-tuned) on the test set."""
    eval_cfg = EvalConfig()
    model_cfg = ModelConfig(use_4bit=args.use_4bit)
    gen_cfg = GenerationConfig()
    data_cfg = DataConfig()

    system_prompt = get_prompt(args.prompt)
    label = "base model" if not args.lora else f"fine-tuned ({args.lora})"
    print(f"Evaluating {label} with prompt '{args.prompt}'")

    # Load test set
    splits = get_dataset_splits(data_cfg.dataset_name)
    test_ds = splits["test"]
    if args.max_samples:
        test_ds = test_ds.select(range(min(args.max_samples, len(test_ds))))
    print(f"Test set: {len(test_ds)} samples")

    # Generate responses
    responses = generate_responses(
        test_ds, system_prompt, args.backend,
        model_cfg, gen_cfg, data_cfg, lora_path=args.lora,
    )

    # Score with GPT-4o
    print("Scoring responses with GPT-4o...")
    results = score_responses(responses, eval_cfg)
    print_results(results, label)

    if args.output:
        _save_results(results, args.output, label)
    else:
        default_name = "ft_results.json" if args.lora else "base_results.json"
        default_path = os.path.join(eval_cfg.eval_output_dir, default_name)
        _save_results(results, default_path, label)


def cmd_score_reference(args) -> None:
    """Score the reference responses from the dataset itself."""
    eval_cfg = EvalConfig()
    data_cfg = DataConfig()

    print(f"Scoring reference column '{args.column}' from dataset...")

    # Load test set
    splits = get_dataset_splits(data_cfg.dataset_name)
    test_ds = splits["test"]
    if args.max_samples:
        test_ds = test_ds.select(range(min(args.max_samples, len(test_ds))))
    print(f"Test set: {len(test_ds)} samples")

    # Build response list from reference column
    responses = []
    for row in test_ds:
        responses.append({
            "user_prompt": row["user"],
            "task": row["task"],
            "evaluation_metric": row["evaluation_metric"],
            "response": row[args.column],
        })

    # Score with GPT-4o
    print("Scoring responses with GPT-4o...")
    results = score_responses(responses, eval_cfg)
    print_results(results, f"reference ({args.column})")

    output_path = args.output or os.path.join(
        eval_cfg.eval_output_dir, f"reference_{args.column}.json"
    )
    _save_results(results, output_path, f"reference ({args.column})")


def cmd_compare(args) -> None:
    """Compare base model vs fine-tuned model on the test set."""
    eval_cfg = EvalConfig()
    model_cfg = ModelConfig(use_4bit=args.use_4bit)
    gen_cfg = GenerationConfig()
    data_cfg = DataConfig()

    system_prompt = get_prompt(args.prompt)
    print(f"Comparing base vs fine-tuned model with prompt '{args.prompt}'")

    # Load test set
    splits = get_dataset_splits(data_cfg.dataset_name)
    test_ds = splits["test"]
    if args.max_samples:
        test_ds = test_ds.select(range(min(args.max_samples, len(test_ds))))
    print(f"Test set: {len(test_ds)} samples")

    # --- Base model ---
    print("\n--- Base Model ---")
    base_responses = generate_responses(
        test_ds, system_prompt, args.backend,
        model_cfg, gen_cfg, data_cfg, lora_path=None,
    )
    print("Scoring base model responses with GPT-4o...")
    base_results = score_responses(base_responses, eval_cfg)
    print_results(base_results, "Base Model")

    # --- Fine-tuned model ---
    print("\n--- Fine-tuned Model ---")
    ft_responses = generate_responses(
        test_ds, system_prompt, args.backend,
        model_cfg, gen_cfg, data_cfg, lora_path=args.lora,
    )
    print("Scoring fine-tuned model responses with GPT-4o...")
    ft_results = score_responses(ft_responses, eval_cfg)
    print_results(ft_results, f"Fine-tuned ({args.lora})")

    # --- Comparison ---
    print_comparison(base_results, ft_results)

    # Save results
    output_dir = args.output or eval_cfg.eval_output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _save_results(base_results, os.path.join(output_dir, "compare_base.json"), "Base Model")
    _save_results(ft_results, os.path.join(output_dir, "compare_ft.json"), f"Fine-tuned ({args.lora})")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate reward-hacking behavior in base vs fine-tuned models"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- score-model ---
    sp_model = subparsers.add_parser(
        "score-model", help="Score a model (base or fine-tuned) on the test set"
    )
    sp_model.add_argument("--lora", type=str, default=None, help="Path to LoRA adapter (omit for base model)")
    sp_model.add_argument("--backend", choices=["vllm", "hf"], default="vllm", help="Inference backend")
    sp_model.add_argument("--prompt", type=str, default="neutral",
                          help=f"System prompt key. Available: {', '.join(PROMPTS.keys())}")
    sp_model.add_argument("--max-samples", type=int, default=None, help="Limit number of test samples")
    sp_model.add_argument("--output", type=str, default=None, help="Output JSON file path")
    sp_model.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization (HF backend only)")
    sp_model.set_defaults(func=cmd_score_model)

    # --- score-reference ---
    sp_ref = subparsers.add_parser(
        "score-reference", help="Score reference responses from the dataset"
    )
    sp_ref.add_argument("--column", choices=["school_of_reward_hacks", "control"],
                        required=True, help="Which reference column to score")
    sp_ref.add_argument("--max-samples", type=int, default=None, help="Limit number of test samples")
    sp_ref.add_argument("--output", type=str, default=None, help="Output JSON file path")
    sp_ref.set_defaults(func=cmd_score_reference)

    # --- compare ---
    sp_cmp = subparsers.add_parser(
        "compare", help="Compare base vs fine-tuned model side-by-side"
    )
    sp_cmp.add_argument("--lora", type=str, required=True, help="Path to LoRA adapter for fine-tuned model")
    sp_cmp.add_argument("--backend", choices=["vllm", "hf"], default="vllm", help="Inference backend")
    sp_cmp.add_argument("--prompt", type=str, default="neutral",
                        help=f"System prompt key. Available: {', '.join(PROMPTS.keys())}")
    sp_cmp.add_argument("--max-samples", type=int, default=None, help="Limit number of test samples")
    sp_cmp.add_argument("--output", type=str, default=None, help="Output directory for comparison results")
    sp_cmp.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization (HF backend only)")
    sp_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
