"""
Loads and runs inference with the language model.
"""

import json
import logging
import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from prompts import SYSTEM_PROMPT, BASE_MODEL, ADAPTER_REPO

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None
_lock = threading.Lock()


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

        logger.info(f"Loading base model: {BASE_MODEL}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )

        logger.info(f"Loading LoRA adapter: {ADAPTER_REPO}")
        _model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
        _model.eval()

        _tokenizer = AutoTokenizer.from_pretrained(ADAPTER_REPO)

        logger.info("Model loaded successfully!")
        return _model, _tokenizer


def run_inference(
    user_query: str,
    tool_docs: str,
    history: list = None,
    max_new_tokens: int = 512
) -> str:
    """
    Generates a text response from the model based on the query and tool specifications.
    """
    if history is None:
        history = []

    model, tokenizer = load_model()

    system_text = SYSTEM_PROMPT.format(tool_docs=tool_docs.strip())
    user_text = f"Information: {json.dumps(history)}\nQuery: {user_query.strip()}"

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]

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

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][prompt_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response
