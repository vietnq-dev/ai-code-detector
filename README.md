- **9 Mar**: Download fine-tuned checkpoint (CodeBERT) [here](https://drive.google.com/drive/folders/1wKwwgXW_pC65XJOlL_ogSyxP1Sr9cI26?usp=sharing) (0.34 on leaderboard test).

# SemEval-2026 Task 13 — Detecting Machine-Generated Code

Fine-tune `microsoft/codebert-base` for  Subtask A

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

Place official `.parquet` files under `data/task_<x>/`.

The loader auto-discovers files by keyword (`train`, `validation`, `test`)
so the organiser naming convention works out of the box:

```text
data/task_a/
  task_a_training_set_1.parquet
  task_a_validation_set.parquet
  task_a_test_set_sample.parquet
data/task_b/
  ...
data/task_c/
  ...
```

Each parquet must have at least a `code` column and a `label` column
(integer label ID). Test files should include an `id` column.

## 3) Training

```bash
# Subtask A (binary)
uv run python train.py --task subtask_a

# Subtask B (11-class)
uv run python train.py --task subtask_b

# Subtask C (4-class)
uv run python train.py --task subtask_c
```

Checkpoints are saved to `checkpoints/<task>/` (best model in
`checkpoints/<task>/best/`).

## 4) Prediction & Submission

```bash
uv run python predict.py \
  --task subtask_a \
  --checkpoint checkpoints/subtask_a/best \
  --test-file data/task_a/task_a_test.parquet \
  --output artifacts/subtask_a/submission.csv
```

Writes `artifacts/subtask_a/submission.csv` with columns `id,label`.

Override paths as needed:

```bash
uv run python predict.py \
    --task subtask_b \
    --checkpoint checkpoints/subtask_b/best \
    --test-file data/raw/subtask_b/test.parquet \
    --output artifacts/subtask_b/submission.csv \
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
│   │   └── codebert-base.yaml  # Model & training defaults
│   └── tasks/
│       ├── subtask_a.yaml
│       ├── subtask_b.yaml
│       └── subtask_c.yaml
├── data/task_{a,b,c}/          # Official .parquet files
├── src/semeval2026_task13/
│   ├── data/
│   │   └── dataset.py          # Parquet loading & tokenization
│   ├── models/
│   │   └── classifier.py       # CodeBERT model builder
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
