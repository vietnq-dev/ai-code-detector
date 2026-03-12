"""Kaggle submission CSV generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def generate_submission(
    ids: list[int] | np.ndarray,
    predictions: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Write a two-column ``id,label`` CSV expected by the Kaggle scorer.

    Args:
        ids: Sample identifiers (same order as predictions).
        predictions: Predicted integer label IDs.
        output_path: Destination CSV path.

    Returns:
        Resolved path of the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"id": ids, "label": predictions})
    df.to_csv(output_path, index=False)

    logger.info("Submission saved to {} ({} rows)", output_path, len(df))
    return output_path
