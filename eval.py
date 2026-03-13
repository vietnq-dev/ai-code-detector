#!/usr/bin/env python
"""Evaluate a saved checkpoint on the validation split."""

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
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DataCollatorWithPadding

from semeval2026_task13.data.dataset import load_splits, tokenize_dataset
from semeval2026_task13.models.classifier import build_tokenizer, get_device, load_model_for_inference

# Quiet noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)


def setup_logging(task_name: str) -> Path:
    """Configure console + file logging and return log path."""
    log_dir = Path("logs") / task_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "eval.log"

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
    p.add_argument("--data-dir", default=None, help="Override data directory.")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes.")
    return p.parse_args()


def _extract_logits(model_output: object) -> np.ndarray:
    if isinstance(model_output, dict):
        logits = model_output.get("logits")
    elif hasattr(model_output, "logits"):
        logits = model_output.logits
    elif isinstance(model_output, (tuple, list)):
        logits = model_output[0] if model_output else model_output
    else:
        logits = model_output
    return np.asarray(logits)


def main() -> None:
    """Entry point: load checkpoint -> evaluate validation split."""
    args = parse_args()
    log_path = setup_logging(args.task)
    logger.info("Logging to {}", log_path)

    device = get_device()
    logger.info("Selected device: {}", device)

    # Resolve data directory from task config
    task_cfg_path = Path(f"configs/tasks/{args.task}.yaml")
    with open(task_cfg_path) as fh:
        task_cfg = yaml.safe_load(fh) or {}

    data_dir = Path(args.data_dir or task_cfg.get("data_dir", f"data/raw/{args.task}"))

    logger.info("Loading splits from {}", data_dir)
    raw_splits = load_splits(data_dir)
    if "validation" not in raw_splits:
        raise RuntimeError("Validation split not found. Provide a validation parquet in data_dir.")

    tokenizer = build_tokenizer(args.checkpoint)
    tokenized = tokenize_dataset(
        hf_datasets.DatasetDict({"validation": raw_splits["validation"]}),
        tokenizer,
        max_length=args.max_length,
    )

    eval_ds = tokenized["validation"]
    if "labels" not in eval_ds.column_names:
        raise RuntimeError("Validation split is missing labels.")

    model_input_columns = [
        name
        for name in ("input_ids", "attention_mask", "token_type_ids", "labels")
        if name in eval_ds.column_names
    ]
    drop_columns = [c for c in eval_ds.column_names if c not in model_input_columns]
    eval_ds = eval_ds.remove_columns(drop_columns)
    eval_ds.set_format(type="torch", columns=model_input_columns)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    dataloader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )

    model = load_model_for_inference(args.checkpoint).to(device)
    model.eval()

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    logger.info("Running evaluation …")
    with torch.inference_mode():
        for batch in tqdm(
            dataloader,
            desc="Evaluating",
            total=len(dataloader),
            unit="batch",
        ):
            labels = batch.pop("labels", None)
            batch = {k: v.to(device, non_blocking=device.type == "cuda") for k, v in batch.items()}
            outputs = model(**batch)
            logits = _extract_logits(outputs)
            preds = np.argmax(logits, axis=-1)
            all_preds.append(preds)
            if labels is not None:
                all_labels.append(labels.cpu().numpy())

    if not all_labels:
        raise RuntimeError("No labels were found during evaluation.")

    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    logger.info("Macro-F1: {:.6f}", macro_f1)
    print(f"Macro-F1: {macro_f1:.6f}")


if __name__ == "__main__":
    main()
