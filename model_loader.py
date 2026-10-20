"""
Loads and runs inference with the language model.
"""

import contextlib
import logging
import threading
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from agent_format import Generation, build_messages
from prompts import BASE_MODEL, ADAPTER_REPO

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None
_lock = threading.Lock()


def _compute_dtype():
    """bf16 where the GPU supports it (A100/L4), fp16 otherwise (e.g. T4)."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model():
    """
    Loads the language model and tokenizer into memory in a thread-safe manner.
    """
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    with _lock:
        if _model is not None:
            return _model, _tokenizer

        dtype = _compute_dtype()
        logger.info(f"Loading base model: {BASE_MODEL} (compute dtype {dtype})")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=dtype,
        )

        logger.info(f"Loading LoRA adapter: {ADAPTER_REPO}")
        _model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
        _model.eval()

        _tokenizer = AutoTokenizer.from_pretrained(ADAPTER_REPO)

        logger.info("Model loaded successfully!")
        return _model, _tokenizer


def generate(messages: list, use_adapter: bool = True, max_new_tokens: int = 512) -> Generation:
    """
    Greedily generates a response for chat messages. With `use_adapter=False` the LoRA
    adapter is disabled, giving the base model's output from the same loaded weights.
    """
    model, tokenizer = load_model()

    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    )

    if hasattr(encoded, "input_ids"):
        input_ids = encoded["input_ids"].to(model.device)
    else:
        input_ids = encoded.to(model.device)

    prompt_len = input_ids.shape[-1]
    adapter_context = contextlib.nullcontext() if use_adapter else model.disable_adapter()

    start = time.perf_counter()
    with torch.no_grad(), adapter_context:
        output_ids = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - start) * 1000

    new_tokens = output_ids[0][prompt_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return Generation(response, int(new_tokens.shape[-1]), latency_ms)


def run_inference(
    user_query: str,
    tool_docs: str,
    history: list = None,
    max_new_tokens: int = 512
) -> Generation:
    """
    Generates a response from the model based on the query and tool specifications.
    """
    messages = build_messages(tool_docs, user_query, history or [])
    return generate(messages, max_new_tokens=max_new_tokens)
