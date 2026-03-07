"""CodeBERT sequence classification model builder with LoRA + quantization."""

from __future__ import annotations

import torch
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from semeval2026_task13.utils.config import ExperimentConfig


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


def _make_bnb_config() -> BitsAndBytesConfig:
    """Create a 4-bit NF4 quantization config."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def build_model(config: ExperimentConfig) -> PreTrainedModel:
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
        A ``PreTrainedModel`` ready for the HF ``Trainer``.
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
