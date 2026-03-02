"""Central configuration for the reward-hacking fine-tuning experiment."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen3-8B"
    torch_dtype: str = "bfloat16"
    # Set to True to use 4-bit quantization during generation (saves VRAM)
    use_4bit: bool = False


@dataclass
class GenerationConfig:
    """Qwen3 recommended sampling params for thinking mode."""
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_new_tokens: int = 32768
    # How many completions to generate per prompt (for rejection sampling)
    num_completions: int = 1
    # vLLM tensor parallel size (number of GPUs)
    tensor_parallel_size: int = 1


@dataclass
class DataConfig:
    dataset_name: str = "longtermrisk/school-of-reward-hacks"
    output_dir: str = "data/generated"
    # Which column to use as the "desired" response style for the system prompt
    # "school_of_reward_hacks" = reward-hacking responses
    # "control" = good-faith responses
    target_column: str = "school_of_reward_hacks"


@dataclass
class TrainingConfig:
    output_dir: str = "checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    logging_steps: int = 10
    save_strategy: str = "epoch"
    # LoRA parameters
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
    )
    # Max sequence length for training
    max_seq_length: int = 4096
    # Gradient checkpointing to save memory
    gradient_checkpointing: bool = True
    # W&B project name (set to None to disable)
    wandb_project: str | None = "qwen3-reward-hack-sft"
