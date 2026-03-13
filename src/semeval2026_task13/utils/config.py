"""Experiment configuration loaded from YAML files."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


@dataclass
class ExperimentConfig:
    """Unified configuration merging model and task YAML settings.

    Attributes:
        model_name: HuggingFace model identifier.
        max_length: Maximum tokenizer sequence length.
        task_name: Subtask identifier (subtask_a / subtask_b / subtask_c).
        num_labels: Number of classification labels.
        data_dir: Directory containing train/test parquet files.
        learning_rate: AdamW learning rate.
        per_device_train_batch_size: Training batch size per device.
        per_device_eval_batch_size: Evaluation batch size per device.
        num_train_epochs: Total training epochs.
        max_steps: Optional cap on total training steps.
        weight_decay: AdamW weight decay.
        warmup_ratio: Fraction of steps for LR warmup.
        gradient_accumulation_steps: Gradient accumulation steps.
        fp16: Whether to use mixed-precision training.
        seed: Global random seed.
        use_lora: Enable LoRA adapters for parameter-efficient fine-tuning.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout in LoRA layers.
        lora_target_modules: Module names to adapt with LoRA. When empty,
            the model builder will infer architecture-specific defaults.
        quantize_4bit: Load base model in 4-bit (CUDA only, skipped elsewhere).
        gradient_checkpointing: Trade compute for memory by checkpointing
            activations.
        eval_strategy: Evaluation cadence for the Trainer.
        eval_steps: Number of steps between evaluations when using step-based
            evaluation.
        save_strategy: Checkpoint save cadence for the Trainer.
        save_steps: Number of steps between checkpoint saves when using
            step-based saving.
        save_total_limit: Maximum number of checkpoints to retain on disk.
        load_best_model_at_end: Whether to restore the best checkpoint after
            training finishes.
        dataloader_num_workers: Number of worker processes for data loading.
        dataloader_pin_memory: Whether to pin CPU memory for faster GPU transfer.
        group_by_length: Whether to bucket samples by sequence length to
            reduce padding waste.
        output_dir: Root directory for checkpoints.
        log_dir: Root directory for TensorBoard / training logs.
        run_dir: Optional resolved run directory (set at runtime).
    """

    # Model
    model_name: str = "project-droid/DroidDetect-Base"
    max_length: int = 256

    # Task
    task_name: str = "subtask_a"
    num_labels: int = 2
    data_dir: str = "data/task_a"

    # Training hyper-parameters
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 32
    num_train_epochs: int = 5
    max_steps: int | None = None
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    seed: int = 42

    # Memory optimisation
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=list)
    quantize_4bit: bool = False
    gradient_checkpointing: bool = False
    eval_strategy: str = "epoch"
    eval_steps: int | None = None
    save_strategy: str = "epoch"
    save_steps: int | None = None
    save_total_limit: int = 2
    load_best_model_at_end: bool = True

    # Data loading
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    group_by_length: bool = True

    # Output paths
    output_dir: str = "checkpoints"
    log_dir: str = "logs"
    run_dir: str | None = None

    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        model_config_path: str | Path,
        task_config_path: str | Path,
        **overrides: Any,
    ) -> ExperimentConfig:
        """Create config by merging a model YAML, a task YAML, and CLI overrides.

        Args:
            model_config_path: Path to the model YAML file.
            task_config_path: Path to the task YAML file.
            **overrides: Keyword arguments that take highest priority.

        Returns:
            A fully-resolved ``ExperimentConfig`` instance.
        """
        merged: dict[str, Any] = {}

        for path in (model_config_path, task_config_path):
            with open(path) as fh:
                data = yaml.safe_load(fh) or {}
            merged.update(data)

        merged.update({k: v for k, v in overrides.items() if v is not None})

        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in merged.items() if k in valid_fields}

        config = cls(**filtered)
        logger.info("Loaded config: {}", config)
        return config
