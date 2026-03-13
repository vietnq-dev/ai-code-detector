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
    preds_raw, labels = eval_pred
    if isinstance(preds_raw, dict):
        logits = preds_raw.get("logits")
    elif isinstance(preds_raw, (tuple, list)):
        logits = preds_raw[0] if preds_raw else preds_raw
    else:
        logits = preds_raw

    logits = np.asarray(logits)
    labels = np.asarray(labels)
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": float(f1_score(labels, preds, average="macro"))}
