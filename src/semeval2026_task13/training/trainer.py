"""HuggingFace Trainer factory driven by ``ExperimentConfig``."""

from __future__ import annotations

from pathlib import Path

import datasets as hf_datasets
import torch
from loguru import logger
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from semeval2026_task13.evaluation.metrics import compute_metrics
from semeval2026_task13.utils.config import ExperimentConfig


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
    output_dir = Path(config.output_dir) / config.task_name
    log_dir = Path(config.log_dir) / config.task_name
    precision = _resolve_precision(config.fp16)

    gc_kwargs = {"use_reentrant": False} if config.gradient_checkpointing else None

    return TrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(log_dir),
        # Optimiser
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        # Batching
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        # Precision & reproducibility
        fp16=precision["fp16"],
        bf16=precision["bf16"],
        seed=config.seed,
        # Memory
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs=gc_kwargs,
        # Evaluation & checkpointing
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        # Logging
        logging_steps=50,
        report_to="none",
    )


def build_trainer(
    model: PreTrainedModel,
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

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
