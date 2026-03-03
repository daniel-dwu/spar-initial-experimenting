# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Reward-hacking fine-tuning experiment studying whether fine-tuning Qwen3-8B on its own chain-of-thought completions increases reward-hacking behavior. Uses `longtermrisk/school-of-reward-hacks` dataset (1,073 rows).

## Architecture

Three-stage pipeline, each stage is a standalone script:

```text
generate_data.py → train.py → evaluate.py
```

1. **generate_data.py** — Loads dataset (train split), generates Qwen3-8B completions with CoT (`<think>...</think>`), saves JSONL. Supports vLLM (fast, batched) and HF transformers (slower, less VRAM) backends.
2. **train.py** — LoRA SFT on generated completions using TRL's SFTTrainer. Trains on full sequence including thinking tokens. System prompt at training time can differ from generation ("recontextualization").
3. **evaluate.py** — Generates responses on held-out test set, scores them with GPT-4o as judge (0-10 on task's stated metric). Three subcommands: `score-model`, `score-reference`, `compare`.

**config.py** is the single source of truth: all dataclasses (`ModelConfig`, `GenerationConfig`, `DataConfig`, `TrainingConfig`, `EvalConfig`), the `PROMPTS` dict, `ExperimentConfig` presets, `JUDGE_SYSTEM_PROMPT`, and `get_dataset_splits()`.

## Key Design Decisions

- **Recontextualization**: The system prompt used during training can differ from generation. Prompts: `neutral`, `negative`, `permissive`, `metric_aware`, `strongly_permissive`.
- **Train/test split**: 85/15, seed 42. Both `generate_data.py` and `evaluate.py` call `get_dataset_splits()` to guarantee the same split.
- **CoT parsing**: `split_thinking()` in `generate_data.py` parses `<think>...</think>` tags. Imported by `evaluate.py`.
- **LoRA**: Targets all linear layers (q/k/v/o/gate/up/down_proj), r=64, alpha=128.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate training data (train split only by default)
python generate_data.py --backend vllm --prompt neutral
python generate_data.py --backend hf --use-4bit --max-samples 50  # low-VRAM test

# Fine-tune
python train.py --data data/generated/completions.jsonl --prompt permissive

# Evaluate
python evaluate.py score-reference --column control --max-samples 5
python evaluate.py score-model --backend hf --prompt neutral --max-samples 5
python evaluate.py compare --lora checkpoints/final --backend hf --max-samples 10
```
