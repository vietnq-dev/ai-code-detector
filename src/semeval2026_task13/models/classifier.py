"""CodeBERT sequence classification model builder with LoRA + quantization."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
)
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from semeval2026_task13.utils.config import ExperimentConfig


def _is_droiddetect_model(model_name: str) -> bool:
    return "droiddetect" in model_name.lower()


def _load_droiddetect_backbone(model_name: str, **kwargs: dict) -> nn.Module:
    try:
        return AutoModel.from_pretrained(model_name, trust_remote_code=True, **kwargs)
    except Exception as exc:
        logger.warning(
            "Failed to load DroidDetect encoder from {} ({}). Falling back to answerdotai/ModernBERT-base",
            model_name,
            exc,
        )
        return AutoModel.from_pretrained(
            "answerdotai/ModernBERT-base",
            trust_remote_code=True,
            **kwargs,
        )


def _masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return last_hidden_state.mean(dim=1)

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def _batch_hard_soft_margin_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    labels = labels.view(-1)
    if embeddings.shape[0] < 2:
        return embeddings.new_zeros(())

    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
    same_label = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    same_label.fill_diagonal_(False)
    diff_label = ~same_label
    diff_label.fill_diagonal_(False)

    hardest_pos = pairwise_dist.masked_fill(~same_label, 0.0).max(dim=1).values
    hardest_neg = pairwise_dist.masked_fill(~diff_label, float("inf")).min(dim=1).values

    valid = same_label.any(dim=1) & diff_label.any(dim=1)
    if not valid.any():
        return embeddings.new_zeros(())

    margin_logits = hardest_pos[valid] - hardest_neg[valid]
    return F.softplus(margin_logits).mean()


class DroidDetectTLModel(nn.Module):
    """Binary DroidDetect classifier using CE + batch-hard triplet loss."""

    CONFIG_FILENAME = "droiddetect_tl_config.json"
    WEIGHTS_FILENAME = "pytorch_model.bin"

    def __init__(
        self,
        text_encoder: nn.Module,
        source_model_name: str,
        base_model_name: str,
        num_classes: int = 2,
        projection_dim: int = 128,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.source_model_name = source_model_name
        self.base_model_name = base_model_name
        self.num_classes = num_classes
        hidden_size = int(getattr(text_encoder.config, "hidden_size", 768))

        self.text_projection = nn.Linear(hidden_size, projection_dim)
        self.classifier = nn.Linear(projection_dim, num_classes)
        self.class_weights = class_weights

    def forward(
        self,
        labels: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        **_: dict,
    ) -> dict[str, torch.Tensor]:
        encoder_inputs: dict[str, torch.Tensor] = {}
        if input_ids is not None:
            encoder_inputs["input_ids"] = input_ids
        if attention_mask is not None:
            encoder_inputs["attention_mask"] = attention_mask
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids

        outputs = self.text_encoder(**encoder_inputs)
        sentence_embeddings = _masked_mean_pool(outputs.last_hidden_state, attention_mask)
        projected_text = F.relu(self.text_projection(sentence_embeddings))
        logits = self.classifier(projected_text)

        result: dict[str, torch.Tensor] = {
            "logits": logits,
            "fused_embedding": projected_text,
        }

        if labels is None:
            return result

        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        ce_loss = nn.CrossEntropyLoss(weight=weight)(
            logits.view(-1, self.num_classes),
            labels.view(-1),
        )
        triplet_loss = _batch_hard_soft_margin_triplet_loss(projected_text, labels)
        total_loss = ce_loss + (0.1 * triplet_loss)

        result["loss"] = total_loss
        result["cross_entropy_loss"] = ce_loss
        result["contrastive_loss"] = triplet_loss
        return result

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), save_directory / self.WEIGHTS_FILENAME)
        with open(save_directory / self.CONFIG_FILENAME, "w") as fh:
            json.dump(
                {
                    "source_model_name": self.source_model_name,
                    "base_model_name": self.base_model_name,
                    "num_classes": self.num_classes,
                    "projection_dim": self.text_projection.out_features,
                },
                fh,
                indent=2,
            )

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> DroidDetectTLModel:
        model_path = Path(model_path)
        config_path = model_path / cls.CONFIG_FILENAME
        if not config_path.exists():
            raise FileNotFoundError(f"Missing {cls.CONFIG_FILENAME} in {model_path}")

        with open(config_path) as fh:
            cfg = json.load(fh)

        base_model_name = str(
            cfg.get(
                "base_model_name",
                cfg.get("source_model_name", "answerdotai/ModernBERT-base"),
            )
        )
        text_encoder = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
        model = cls(
            text_encoder=text_encoder,
            source_model_name=str(cfg.get("source_model_name", base_model_name)),
            base_model_name=base_model_name,
            num_classes=int(cfg.get("num_classes", 2)),
            projection_dim=int(cfg.get("projection_dim", 128)),
        )
        state_dict = torch.load(model_path / cls.WEIGHTS_FILENAME, map_location="cpu")
        model.load_state_dict(state_dict)
        return model


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
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    logger.info("Tokenizer loaded: {} (vocab={})", model_name, tokenizer.vocab_size)
    return tokenizer


def _make_bnb_config() -> BitsAndBytesConfig:
    """Create a 4-bit NF4 quantization config."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def build_model(config: ExperimentConfig) -> nn.Module:
    """Build a classification model with optional LoRA and 4-bit quantization.

    Pipeline:
        1. Load base model (quantized on CUDA if ``config.quantize_4bit``).
        2. Prepare for k-bit training when quantized.
        3. Wrap with LoRA adapters when ``config.use_lora``.

    After training the returned model can be merged back into a standard
    HuggingFace model via ``model.merge_and_unload()``.

    Args:
        config: Experiment configuration.

    Returns:
        A model ready for the HF ``Trainer``.
    """
    # --- quantization (CUDA-only) ----------------------------------------
    bnb_config = None
    if config.quantize_4bit:
        if torch.cuda.is_available():
            bnb_config = _make_bnb_config()
            logger.info("4-bit NF4 quantization enabled (CUDA)")
        else:
            logger.warning(
                "quantize_4bit=True but no CUDA device — skipping quantization"
            )

    if _is_droiddetect_model(config.model_name):
        if config.num_labels != 2:
            raise ValueError(
                f"DroidDetect-Base path expects 2 labels, got num_labels={config.num_labels}"
            )

        encoder_kwargs: dict = {}
        if bnb_config is not None:
            encoder_kwargs["quantization_config"] = bnb_config
        if bnb_config is not None:
            encoder_kwargs["device_map"] = "auto"

        text_encoder = _load_droiddetect_backbone(config.model_name, **encoder_kwargs)
        base_model_name = str(getattr(text_encoder.config, "_name_or_path", config.model_name))
        model = DroidDetectTLModel(
            text_encoder=text_encoder,
            source_model_name=config.model_name,
            base_model_name=base_model_name,
            num_classes=2,
            projection_dim=128,
        )
        logger.info("Loaded DroidDetect TL model: {} | labels=2", config.model_name)

        if config.use_lora:
            logger.warning("LoRA is not applied for DroidDetect TL model; set use_lora=false")

        return model

    # --- base model -------------------------------------------------------
    load_kwargs: dict = {
        "num_labels": config.num_labels,
        "problem_type": "single_label_classification",
        "quantization_config": bnb_config,
    }
    if bnb_config is not None:
        # Ensures quantized modules are placed on GPU and bnb state is initialized.
        load_kwargs["device_map"] = "auto"

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        **load_kwargs,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Base model loaded: {} | labels={} | params={:,}",
        config.model_name,
        config.num_labels,
        n_params,
    )

    # --- prepare for quantized training -----------------------------------
    if bnb_config is not None:
        # For RoBERTa classifiers, wrapping a 4-bit classification head with
        # PEFT modules_to_save can trigger bitsandbytes assertion errors.
        # Replace it with a standard fp32 head to keep training stable.
        if getattr(model.config, "model_type", "") == "roberta" and hasattr(model, "classifier"):
            model.classifier = RobertaClassificationHead(model.config).to(model.device)
            logger.info("Replaced quantized classifier head with fp32 head for PEFT compatibility")

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )

    # --- LoRA -------------------------------------------------------------
    if config.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["query", "value"],
        )
        model = get_peft_model(model, lora_config)

        trainable, total = model.get_nb_trainable_parameters()
        logger.info(
            "LoRA applied — trainable: {:,} / {:,} ({:.2%})",
            trainable,
            total,
            trainable / total,
        )

    return model


def load_model_for_inference(model_path: str | Path) -> nn.Module:
    """Load a trained checkpoint for inference (CodeBERT or DroidDetect TL)."""
    model_path = Path(model_path)
    droid_cfg = model_path / DroidDetectTLModel.CONFIG_FILENAME

    if droid_cfg.exists():
        logger.info("Detected DroidDetect TL checkpoint: {}", model_path)
        return DroidDetectTLModel.from_pretrained(model_path)

    logger.info("Loading HuggingFace sequence classification checkpoint: {}", model_path)
    return AutoModelForSequenceClassification.from_pretrained(str(model_path))
