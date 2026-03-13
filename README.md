# SemEval-2026 Task 13 — Detecting Machine-Generated Code

This project trains **binary AI-code detection** models (`num_labels: 2`) using either:

- `microsoft/codebert-base`
- `project-droid/DroidDetect-Base` (with CE + batch-hard triplet objective)

## 1) Installation

Requires **Python 3.10+** and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## 2) Data

Download data:

```
curl -L -o semeval-2026-task13.zip\
  https://www.kaggle.com/api/v1/datasets/download/daniilor/semeval-2026-task13
```

Place official `.parquet` files under `data/task_a/`.

The loader auto-discovers files by keyword (`train`, `validation`, `test`)
so the organiser naming convention works out of the box:

```text
data/task_a/
  task_a_training_set_1.parquet
  task_a_validation_set.parquet
  task_a_test_set_sample.parquet
```

Each parquet must have at least a `code` column and a `label` column
(integer label ID). Test files should include an `ID` column.

## 3) Training

This repository is configured for **binary detection** (`num_labels: 2`).

```bash
# Binary training with CodeBERT (default)
uv run python train.py --task subtask_a

# Binary training with DroidDetect-Base + TL head
uv run python train.py --task subtask_a --model-config configs/model/droiddetect-base.yaml
```

Checkpoints are saved to `checkpoints/<task>/` (best model in
`checkpoints/<task>/best/`).

## 4) Prediction & Submission

```bash
uv run python predict.py \
  --task subtask_a \
  --checkpoint checkpoints/subtask_a/DroidDetect-Base-202603130715/checkpoint-15625 \
  --test-file data/task_a/task_a_test.parquet \
  --output artifacts/subtask_a/submission-droid-detect.csv
```

Writes `artifacts/subtask_a/submission.csv` with columns `id,label`.

Override paths as needed:

```bash
uv run python predict.py \
  --task subtask_a \
  --checkpoint checkpoints/subtask_a/best \
  --test-file data/task_a/task_a_test.parquet \
  --output artifacts/subtask_a/submission-droid-detect.csv \
    --batch-size 64
```

## 5) Project Layout

```text
.
├── train.py                    # Training entry point
├── predict.py                  # Inference + submission CSV
├── pyproject.toml
├── configs/
│   ├── model/
│   │   ├── codebert-base.yaml
│   │   └── droiddetect-base.yaml
├── data/task_a/                # Official .parquet files
├── src/semeval2026_task13/
│   ├── data/
│   │   └── dataset.py          # Parquet loading & tokenization
│   ├── models/
│   │   └── classifier.py       # CodeBERT + DroidDetect model builder
│   ├── training/
│   │   └── trainer.py          # HF Trainer factory
│   ├── evaluation/
│   │   └── metrics.py          # Macro-F1 metric
│   └── utils/
│       ├── config.py           # YAML config loader
│       └── submission.py       # Kaggle CSV exporter
├── checkpoints/                # Saved models
├── artifacts/                  # Submission CSVs
└── logs/                       # Training logs
```