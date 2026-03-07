"""Macro-F1 metric compatible with HuggingFace ``Trainer``."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score
from transformers import EvalPrediction


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Compute macro-averaged F1 score.

    This callable is passed directly to ``Trainer(compute_metrics=...)``.

    Args:
        eval_pred: Named tuple with *predictions* (logits) and *label_ids*.

    Returns:
        Dictionary ``{"macro_f1": <score>}``.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": float(f1_score(labels, preds, average="macro"))}
