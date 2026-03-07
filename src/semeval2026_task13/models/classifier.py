"""CodeBERT sequence classification model builder."""

from __future__ import annotations

import torch
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU).

    Returns:
        A ``torch.device`` for the preferred accelerator.
    """
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info("Device: CUDA ({})", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        logger.info("Device: MPS (Apple Silicon)")
    else:
        dev = torch.device("cpu")
        logger.info("Device: CPU")
    return dev


def build_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load a pre-trained tokenizer.

    Args:
        model_name: HuggingFace model identifier
            (e.g. ``microsoft/codebert-base``).

    Returns:
        The corresponding tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("Tokenizer loaded: {} (vocab={})", model_name, tokenizer.vocab_size)
    return tokenizer


def build_model(model_name: str, num_labels: int) -> PreTrainedModel:
    """Instantiate a pre-trained encoder with a classification head.

    Uses ``AutoModelForSequenceClassification`` so the same function
    works for any RoBERTa-family checkpoint (CodeBERT, GraphCodeBERT, …).

    Args:
        model_name: HuggingFace model identifier.
        num_labels: Number of target classes.

    Returns:
        A ``PreTrainedModel`` with a freshly initialised classifier head.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="single_label_classification",
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model loaded: {} | labels={} | params={:,}",
        model_name,
        num_labels,
        n_params,
    )
    return model
