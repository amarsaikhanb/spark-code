"""4-bit quantized Qwen2.5-Coder loading with PEFT/LoRA adapters."""

from __future__ import annotations

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from spark_code.config import Config


def load_model(cfg: Config):
    """Load (model, tokenizer) ready for QLoRA fine-tuning.

    The returned model has trainable LoRA params only; the base weights stay
    frozen at NF4 precision. Tokenizer uses left padding for batched generation.
    """
    print(f"[model] Loading {cfg.model_name}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=cfg.torch_dtype,
        bnb_4bit_use_double_quant=True,
    )
    try:
        import flash_attn  # noqa: F401

        attn_impl = "flash_attention_2"
    except Exception:
        attn_impl = "sdpa"
        print("[model] flash-attn not found; using SDPA")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=cfg.torch_dtype,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_targets,
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    cfg.device = str(next(model.parameters()).device)
    print(f"[model] Device: {cfg.device}")
    return model, tok
