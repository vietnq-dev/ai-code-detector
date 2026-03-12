#!/usr/bin/env python
"""Generate a Kaggle submission CSV from a fine-tuned checkpoint."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import datasets as hf_datasets
import numpy as np
import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from semeval2026_task13.data.dataset import load_parquet, tokenize_dataset
from semeval2026_task13.models.classifier import get_device
from semeval2026_task13.utils.submission import generate_submission

# Quiet noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)


def setup_logging(task_name: str) -> Path:
    """Configure console + file logging and return log path."""
    log_dir = Path("logs") / task_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "predict.log"

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
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes.")
    return p.parse_args()


def main() -> None:
    """Entry point: load checkpoint -> predict test set -> write CSV."""
    args = parse_args()
    log_path = setup_logging(args.task)
    logger.info("Logging to {}", log_path)

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
        all_test_candidates = [p for p in sorted(data_dir.glob("*.parquet")) if "test" in p.stem.lower()]
        if not all_test_candidates:
            raise FileNotFoundError(f"No test parquet found in {data_dir}")

        preferred = data_dir / f"{args.task.replace('subtask_', 'task_')}_test.parquet"
        if preferred.exists():
            test_path = str(preferred)
        else:
            non_sample_candidates = [
                p for p in all_test_candidates if "sample" not in p.stem.lower()
            ]
            selected = non_sample_candidates[0] if non_sample_candidates else all_test_candidates[0]
            test_path = str(selected)

    output_path = args.output or f"artifacts/{args.task}/submission.csv"

    # Load model & tokenizer from merged checkpoint
    logger.info("Loading checkpoint: {}", args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)

    # Load & tokenize test data
    logger.info("Loading test data: {}", test_path)
    test_ds = load_parquet(test_path)

    if "ID" not in test_ds.column_names:
        raise ValueError(
            f"Missing required 'id' column in test parquet: {test_path}. "
            "Submission IDs must match the test file IDs."
        )
    ids = test_ds["ID"]

    tokenized = tokenize_dataset(
        hf_datasets.DatasetDict({"test": test_ds}),
        tokenizer,
        max_length=args.max_length,
    )

    # Run inference with a manual DataLoader loop to reduce Trainer overhead.
    model = model.to(device)
    model.eval()

    test_tokenized = tokenized["test"]
    model_input_columns = [
        name
        for name in ("input_ids", "attention_mask", "token_type_ids")
        if name in test_tokenized.column_names
    ]
    drop_columns = [c for c in test_tokenized.column_names if c not in model_input_columns]
    inference_ds = test_tokenized.remove_columns(drop_columns)
    inference_ds.set_format(type="torch", columns=model_input_columns)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    dataloader = DataLoader(
        inference_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )

    logger.info("Running predictions …")
    pred_batches: list[np.ndarray] = []
    use_autocast = device.type == "cuda"
    with torch.inference_mode():
        for batch in tqdm(
            dataloader,
            desc="Predicting",
            total=len(dataloader),
            unit="batch",
        ):
            batch = {k: v.to(device, non_blocking=use_autocast) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_autocast):
                logits = model(**batch).logits
            pred_batches.append(torch.argmax(logits, dim=-1).cpu().numpy())

    pred_labels = np.concatenate(pred_batches, axis=0)
    if len(ids) != len(pred_labels):
        raise RuntimeError(
            "Prediction count does not match test IDs count: "
            f"{len(pred_labels)} vs {len(ids)}"
        )

    generate_submission(ids, pred_labels, output_path)
    logger.info("Done — submission written to {}", output_path)


if __name__ == "__main__":
    main()
