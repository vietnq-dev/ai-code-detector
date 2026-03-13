"""HuggingFace Trainer factory driven by ``ExperimentConfig``."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import datasets as hf_datasets
import torch
from loguru import logger
from torch import nn
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from semeval2026_task13.evaluation.metrics import compute_metrics
from semeval2026_task13.utils.config import ExperimentConfig


def build_run_dir(config: ExperimentConfig) -> Path:
    """Build a timestamped run directory under the task checkpoint root."""
    if config.run_dir:
        return Path(config.run_dir)
    model_slug = config.model_name.split("/")[-1]
    model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_slug).strip("-")
    stamp = datetime.now().strftime("%Y%m%d%H%M")
    run_name = f"{model_slug}-{stamp}"
    return Path(config.output_dir) / config.task_name / run_name


def _resolve_precision(requested_fp16: bool) -> dict[str, bool]:
    """Pick mixed-precision flags that match the available hardware.

    Args:
        requested_fp16: Whether the user/config requested fp16 training.

    Returns:
        Dict with ``fp16`` and ``bf16`` booleans for ``TrainingArguments``.
    """
    if torch.cuda.is_available():
        logger.info("CUDA detected — fp16={}", requested_fp16)
        return {"fp16": requested_fp16, "bf16": False}

    if torch.backends.mps.is_available():
        logger.info("MPS detected — fp16 disabled (not reliably supported on MPS)")
        return {"fp16": False, "bf16": False}

    logger.info("CPU only — mixed precision disabled")
    return {"fp16": False, "bf16": False}


def build_training_arguments(config: ExperimentConfig) -> TrainingArguments:
    """Translate an ``ExperimentConfig`` into HF ``TrainingArguments``.

    Automatically selects the correct mixed-precision setting based on
    the available accelerator (CUDA -> fp16, MPS / CPU -> disabled) and
    enables gradient checkpointing when requested.

    Args:
        config: Merged experiment configuration.

    Returns:
        Fully-populated ``TrainingArguments``.
    """
    output_dir = build_run_dir(config)
    log_dir = Path(config.log_dir) / config.task_name / output_dir.name
    precision = _resolve_precision(config.fp16)
    optim_name = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"

    gc_kwargs = {"use_reentrant": False} if config.gradient_checkpointing else None

    return TrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(log_dir),
        # Optimiser
        optim=optim_name,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        # Batching
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps if config.max_steps is not None else -1,
        # Precision & reproducibility
        fp16=precision["fp16"],
        bf16=precision["bf16"],
        seed=config.seed,
        # Memory
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs=gc_kwargs,
        # Evaluation & checkpointing
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=config.save_total_limit,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        # Logging
        logging_steps=50,
        report_to="none",
        # Data loading
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory and torch.cuda.is_available(),
    )


def build_trainer(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    train_ds: hf_datasets.Dataset,
    eval_ds: hf_datasets.Dataset,
    config: ExperimentConfig,
) -> Trainer:
    """Build a ready-to-run HF ``Trainer``.

    Args:
        model: Classification model (may be a PEFT wrapper).
        tokenizer: Tokenizer (saved alongside the model).
        train_ds: Tokenized training split.
        eval_ds: Tokenized validation split.
        config: Experiment configuration.

    Returns:
        A configured ``Trainer`` instance.
    """
    training_args = build_training_arguments(config)
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    # Convert warmup_ratio -> warmup_steps to avoid deprecation warnings.
    steps_per_epoch = max(
        1,
        (len(train_ds) // config.per_device_train_batch_size)
        // max(1, config.gradient_accumulation_steps),
    )
    total_steps = max(1, int(steps_per_epoch * config.num_train_epochs))
    training_args.warmup_steps = int(total_steps * config.warmup_ratio)
    training_args.warmup_ratio = 0.0

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
