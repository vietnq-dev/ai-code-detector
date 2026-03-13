#!/usr/bin/env python
"""Fine-tune a HuggingFace code model for SemEval-2026 Task 13."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from loguru import logger

from semeval2026_task13.data.dataset import load_splits, tokenize_dataset
from semeval2026_task13.models.classifier import build_model, build_tokenizer, get_device
from semeval2026_task13.training.trainer import build_run_dir, build_trainer
from semeval2026_task13.utils.config import ExperimentConfig

# Quiet noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)


def setup_logging(task_name: str) -> Path:
    """Configure console + file logging and return log path."""
    log_dir = Path("logs") / task_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"

    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )
    logger.add(
        log_path,
        level="INFO",
        rotation="10 MB",
        retention=5,
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )
    return log_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task",
        required=True,
        choices=["subtask_a", "subtask_b", "subtask_c"],
        help="Which subtask to train.",
    )
    p.add_argument(
        "--model-config",
        default="configs/model/codebert-base.yaml",
        help="Path to model YAML config.",
    )
    p.add_argument("--task-config", default=None, help="Override task YAML path.")
    p.add_argument("--data-dir", default=None, help="Override data directory.")
    p.add_argument("--output-dir", default=None, help="Override checkpoint root.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-fp16", action="store_true", help="Disable mixed precision.")
    p.add_argument("--no-lora", action="store_true", help="Disable LoRA (full fine-tune).")
    p.add_argument("--no-quant", action="store_true", help="Disable 4-bit quantization.")
    return p.parse_args()


def main() -> None:
    """Entry point: load config -> data -> model -> train -> save."""
    args = parse_args()

    task_cfg_path = args.task_config or f"configs/tasks/{args.task}.yaml"

    overrides: dict = {}
    if args.data_dir:
        overrides["data_dir"] = args.data_dir
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.epochs is not None:
        overrides["num_train_epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["per_device_train_batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.max_length is not None:
        overrides["max_length"] = args.max_length
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.grad_accum is not None:
        overrides["gradient_accumulation_steps"] = args.grad_accum
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.no_fp16:
        overrides["fp16"] = False
    if args.no_lora:
        overrides["use_lora"] = False
    if args.no_quant:
        overrides["quantize_4bit"] = False

    config = ExperimentConfig.from_yaml(args.model_config, task_cfg_path, **overrides)
    run_dir = build_run_dir(config)
    config.run_dir = str(run_dir)
    log_path = setup_logging(config.task_name)
    logger.info("Logging to {}", log_path)

    # ---- device ----------------------------------------------------------
    device = get_device()
    logger.info("Selected device: {}", device)

    # ---- tokenizer & data ------------------------------------------------
    tokenizer = build_tokenizer(config.model_name)
    raw_splits = load_splits(config.data_dir, seed=config.seed)
    tokenized = tokenize_dataset(raw_splits, tokenizer, max_length=config.max_length)

    # ---- model & trainer -------------------------------------------------
    model = build_model(config)
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_ds=tokenized["train"],
        eval_ds=tokenized["validation"],
        config=config,
    )

    # ---- train -----------------------------------------------------------
    logger.info("Starting training for {}", config.task_name)
    trainer.train()

    # ---- persist best model ----------------------------------------------
    best_dir = Path(config.run_dir) / "best"

    trained_model = trainer.model
    if hasattr(trained_model, "merge_and_unload"):
        logger.info("Merging LoRA adapters into base model …")
        trained_model = trained_model.merge_and_unload()

    trained_model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    logger.info("Best model saved to {}", best_dir)

    metrics = trainer.evaluate()
    logger.info("Final evaluation: {}", metrics)


if __name__ == "__main__":
    main()
