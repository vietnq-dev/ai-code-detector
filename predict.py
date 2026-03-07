#!/usr/bin/env python
"""Generate a Kaggle submission CSV from a fine-tuned checkpoint."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import datasets as hf_datasets
import numpy as np
import yaml
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from semeval2026_task13.data.dataset import load_parquet, tokenize_dataset
from semeval2026_task13.models.classifier import get_device
from semeval2026_task13.utils.submission import generate_submission

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
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the saved model directory (e.g. checkpoints/subtask_a/best).",
    )
    p.add_argument("--test-file", default=None, help="Explicit path to test parquet.")
    p.add_argument("--data-dir", default=None, help="Directory containing test.parquet.")
    p.add_argument("--output", default=None, help="Output CSV path.")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def main() -> None:
    """Entry point: load checkpoint → predict test set → write CSV."""
    args = parse_args()

    device = get_device()
    logger.info("Selected device: {}", device)

    # Resolve data directory from task config
    task_cfg_path = Path(f"configs/tasks/{args.task}.yaml")
    with open(task_cfg_path) as fh:
        task_cfg = yaml.safe_load(fh) or {}

    if args.test_file:
        test_path = args.test_file
    else:
        data_dir = Path(args.data_dir or task_cfg.get("data_dir", f"data/raw/{args.task}"))
        candidates = [p for p in sorted(data_dir.glob("*.parquet")) if "test" in p.stem.lower()]
        if not candidates:
            raise FileNotFoundError(f"No test parquet found in {data_dir}")
        test_path = str(candidates[0])

    output_path = args.output or f"artifacts/{args.task}/submission.csv"

    # Load model & tokenizer from checkpoint
    logger.info("Loading checkpoint: {}", args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)

    # Load & tokenize test data
    logger.info("Loading test data: {}", test_path)
    test_ds = load_parquet(test_path)

    ids = test_ds["id"] if "id" in test_ds.column_names else list(range(len(test_ds)))

    tokenized = tokenize_dataset(
        hf_datasets.DatasetDict({"test": test_ds}),
        tokenizer,
        max_length=args.max_length,
    )

    # Run inference
    training_args = TrainingArguments(
        output_dir="/tmp/semeval_predict",
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
    )

    logger.info("Running predictions …")
    output = trainer.predict(tokenized["test"])
    pred_labels = np.argmax(output.predictions, axis=-1)

    generate_submission(ids, pred_labels, output_path)
    logger.info("Done — submission written to {}", output_path)


if __name__ == "__main__":
    main()
