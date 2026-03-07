#!/usr/bin/env python
"""Fine-tune CodeBERT for SemEval-2026 Task 13 (any subtask)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from loguru import logger

from semeval2026_task13.data.dataset import load_splits, tokenize_dataset
from semeval2026_task13.models.classifier import build_model, build_tokenizer, get_device
from semeval2026_task13.training.trainer import build_trainer
from semeval2026_task13.utils.config import ExperimentConfig

# Quiet noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)


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
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-fp16", action="store_true", help="Disable mixed precision.")
    return p.parse_args()


def main() -> None:
    """Entry point: load config → data → model → train → save."""
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
    if args.grad_accum is not None:
        overrides["gradient_accumulation_steps"] = args.grad_accum
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.no_fp16:
        overrides["fp16"] = False

    config = ExperimentConfig.from_yaml(args.model_config, task_cfg_path, **overrides)

    # ---- device ----------------------------------------------------------
    device = get_device()
    logger.info("Selected device: {}", device)

    # ---- tokenizer & data ------------------------------------------------
    tokenizer = build_tokenizer(config.model_name)
    raw_splits = load_splits(config.data_dir, seed=config.seed)
    tokenized = tokenize_dataset(raw_splits, tokenizer, max_length=config.max_length)

    # ---- model & trainer -------------------------------------------------
    model = build_model(config.model_name, config.num_labels)
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
    best_dir = Path(config.output_dir) / config.task_name / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    logger.info("Best model saved to {}", best_dir)

    metrics = trainer.evaluate()
    logger.info("Final evaluation: {}", metrics)


if __name__ == "__main__":
    main()
