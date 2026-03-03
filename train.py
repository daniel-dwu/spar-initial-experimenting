"""
Fine-tune Qwen3-8B on on-policy generated CoT data.

Trains next-token prediction on the full completion (thinking + response),
teaching the model to reproduce both the reasoning traces and the final
answers from its own on-policy generations.

Usage:
    python train.py --data data/generated/completions.jsonl

    # With LoRA (default):
    python train.py --data data/generated/completions.jsonl

    # Full fine-tune (needs much more VRAM):
    python train.py --data data/generated/completions.jsonl --no-lora

    # Resume from checkpoint:
    python train.py --data data/generated/completions.jsonl --resume checkpoints/checkpoint-100
"""

import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

from config import ModelConfig, TrainingConfig, DEFAULT_TRAINING_PROMPT, PROMPTS, get_prompt


def load_generated_data(path: str) -> Dataset:
    """Load the JSONL file produced by generate_data.py."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def format_example(example: dict, tokenizer, system_prompt: str) -> str:
    """Format a single example as a full chat conversation for SFT.

    The training target is the full assistant turn including <think>...</think>
    and the final response, so the model learns both the CoT and the answer.

    The system_prompt here can differ from the one used during generation —
    this is the core of the recontextualisation experiment.
    """
    # Reconstruct the full assistant response with thinking tokens
    thinking = example["thinking"]
    response = example["response"]
    if thinking:
        assistant_content = f"<think>\n{thinking}\n</think>\n{response}"
    else:
        assistant_content = f"<think>\n</think>\n{response}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["user"]},
        {"role": "assistant", "content": assistant_content},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    return text


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-8B on CoT data")
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to completions.jsonl from generate_data.py"
    )
    parser.add_argument(
        "--no-lora", action="store_true",
        help="Full fine-tune instead of LoRA"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from a checkpoint directory"
    )
    parser.add_argument(
        "--prompt", type=str, default=DEFAULT_TRAINING_PROMPT,
        help=f"System prompt key for training (default: {DEFAULT_TRAINING_PROMPT}). "
             f"This can differ from the generation prompt to recontextualise the data. "
             f"Available: {', '.join(PROMPTS.keys())}"
    )
    args = parser.parse_args()

    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()
    if args.no_lora:
        train_cfg.use_lora = False

    system_prompt = get_prompt(args.prompt)
    print(f"Using training prompt '{args.prompt}': {system_prompt[:80]}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and format dataset
    print("Loading generated data...")
    raw_dataset = load_generated_data(args.data)
    print(f"  {len(raw_dataset)} examples loaded")

    # Pre-format all examples with the (possibly different) training prompt
    formatted_texts = [format_example(ex, tokenizer, system_prompt) for ex in raw_dataset]
    train_dataset = Dataset.from_dict({"text": formatted_texts})

    # Load model
    print(f"Loading model {model_cfg.model_name}...")
    load_kwargs = {
        "torch_dtype": getattr(torch, model_cfg.torch_dtype),
        "device_map": "auto",
    }
    if train_cfg.use_lora:
        # For LoRA, optionally load in 4-bit to save memory
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, model_cfg.torch_dtype),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name, **load_kwargs,
    )

    if train_cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Set up LoRA if enabled
    if train_cfg.use_lora:
        lora_config = LoraConfig(
            r=train_cfg.lora_r,
            lora_alpha=train_cfg.lora_alpha,
            lora_dropout=train_cfg.lora_dropout,
            target_modules=train_cfg.lora_target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # W&B setup
    report_to = "wandb" if train_cfg.wandb_project else "none"

    # Training arguments
    sft_config = SFTConfig(
        output_dir=train_cfg.output_dir,
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        warmup_ratio=train_cfg.warmup_ratio,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        bf16=train_cfg.bf16,
        logging_steps=train_cfg.logging_steps,
        save_strategy=train_cfg.save_strategy,
        max_seq_length=train_cfg.max_seq_length,
        dataset_text_field="text",
        report_to=report_to,
        run_name="qwen3-8b-reward-hack-sft" if report_to == "wandb" else None,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    # Save
    print(f"Saving model to {train_cfg.output_dir}/final...")
    trainer.save_model(f"{train_cfg.output_dir}/final")
    tokenizer.save_pretrained(f"{train_cfg.output_dir}/final")
    print("Done!")


if __name__ == "__main__":
    main()
