"""Parquet dataset loading and generic tokenizer pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import datasets as hf_datasets
from loguru import logger
from transformers import PreTrainedTokenizerBase

CODE_COLUMN = "code"
LABEL_COLUMN = "label"

# Columns that are always safe to drop before training.
_META_COLUMNS = {"language", "generator", "source", "problem_id"}


def load_parquet(path: str | Path) -> hf_datasets.Dataset:
    """Load a single parquet file into a HuggingFace ``Dataset``.

    Args:
        path: Path to the ``.parquet`` file.

    Returns:
        A HuggingFace ``Dataset``.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return hf_datasets.Dataset.from_parquet(str(path))


def _find_parquet(data_dir: Path, *keywords: str) -> Path | None:
    """Return the first ``.parquet`` in *data_dir* whose stem contains any keyword."""
    for path in sorted(data_dir.glob("*.parquet")):
        stem = path.stem.lower()
        if any(kw in stem for kw in keywords):
            return path
    return None


def load_splits(
    data_dir: str | Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> hf_datasets.DatasetDict:
    """Load train / validation / test splits from a directory of parquet files.

    File discovery is flexible — any ``.parquet`` whose name contains
    ``train`` / ``validation`` / ``test`` is picked up automatically,
    so both canonical names (``train.parquet``) and organiser names
    (``task_a_training_set_1.parquet``) are supported.

    Args:
        data_dir: Directory containing parquet files.
        val_ratio: Fraction of training data held out for validation when
            no validation file is found.
        seed: Random seed used for the train/val split.

    Returns:
        A ``DatasetDict`` with ``train``, ``validation``, and optionally
        ``test`` splits.

    Raises:
        FileNotFoundError: If no training parquet can be found.
    """
    data_dir = Path(data_dir)
    split_map: dict[str, hf_datasets.Dataset] = {}

    train_path = _find_parquet(data_dir, "train")
    if train_path is None:
        raise FileNotFoundError(f"No training parquet found in {data_dir}")

    logger.info("Train file: {}", train_path.name)
    train_ds = load_parquet(train_path)

    val_path = _find_parquet(data_dir, "validation", "valid", "val", "dev")
    if val_path is not None:
        logger.info("Validation file: {}", val_path.name)
        split_map["train"] = train_ds
        split_map["validation"] = load_parquet(val_path)
    else:
        logger.info(
            "No validation parquet found — holding out {:.0%} of train",
            val_ratio,
        )
        parts = train_ds.train_test_split(test_size=val_ratio, seed=seed)
        split_map["train"] = parts["train"]
        split_map["validation"] = parts["test"]

    test_path = _find_parquet(data_dir, "test")
    if test_path is not None:
        logger.info("Test file: {}", test_path.name)
        split_map["test"] = load_parquet(test_path)

    logger.info("Splits loaded: {}", {k: len(v) for k, v in split_map.items()})
    return hf_datasets.DatasetDict(split_map)


def tokenize_dataset(
    ds: hf_datasets.DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
    code_column: str = CODE_COLUMN,
    label_column: str = LABEL_COLUMN,
) -> hf_datasets.DatasetDict:
    """Tokenize every split and prepare columns expected by HF ``Trainer``.

    * Keeps ``id`` (needed for submission) and ``label`` → renamed to
      ``labels`` for the Trainer.
    * Removes raw text and metadata columns.

    Args:
        ds: A ``DatasetDict`` with one or more splits.
        tokenizer: Pre-trained tokenizer (e.g. ``RobertaTokenizer``).
        max_length: Maximum sequence length.
        code_column: Name of the column containing source code.
        label_column: Name of the column containing integer labels.

    Returns:
        Tokenized ``DatasetDict`` ready for the HF ``Trainer``.
    """

    def _tokenize(examples: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            examples[code_column],
            truncation=True,
            max_length=max_length,
        )

    result: dict[str, hf_datasets.Dataset] = {}

    for split_name, split_ds in ds.items():
        cols_to_remove = [
            c
            for c in split_ds.column_names
            if c in _META_COLUMNS or c == code_column
        ]

        tokenized = split_ds.map(
            _tokenize,
            batched=True,
            remove_columns=cols_to_remove,
            desc=f"Tokenizing {split_name}",
        )

        if label_column in tokenized.column_names:
            tokenized = tokenized.rename_column(label_column, "labels")

        result[split_name] = tokenized

    return hf_datasets.DatasetDict(result)
