"""
Tokenization helpers for foveated VLM training.

Handles chat-template tokenization for all 3 training stages:
  Stage 1 (visual alignment): all-text loss on full chat sequence
  Stage 2 (SFT):              answer-only loss on assistant tokens
  Stage 3 (DPO):              answer-only loss on chosen/rejected

Uses SmolLM2's chat template with an explicit system prompt to avoid
the default "named SmolLM, trained by Hugging Face" injection.

Per-source conditioning prompts match the caption style of each dataset
so the model learns when to produce brief captions vs. detailed narrations.
"""

_tokenizer = None

SYSTEM_PROMPT = "You are a helpful AI assistant."

# Per-source prompts for datasets with empty user field.
SOURCE_PROMPTS = {
    "openvid": "Write a brief caption for this video.",
    "webvid": "What would be the WebVid caption for this video?",
    "vript": "Provide a detailed narration of what happens in this video.",
    "sharegpt4video": "Describe what happens in this video in detail.",
}
DEFAULT_VISUAL_PROMPT = "Describe this."


def get_tokenizer(model_path: str = "/workspace/models/SmolLM2-135M-Instruct"):
    """Load the tokenizer (cached after first call)."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer


def tokenize_stage1(caption: str, tokenizer=None, user_prompt: str = None) -> dict:
    """
    Stage 1 tokenization: all-text loss on full chat-template sequence.

    Parameters
    ----------
    caption : str
        The video caption or text to tokenize.
    tokenizer : optional
        HuggingFace tokenizer. If None, uses cached default.
    user_prompt : str, optional
        Conditioning prompt for the user turn. Defaults to
        "Write a brief caption for this video."

    Returns
    -------
    dict with token_ids (list[int]) and loss_mask (list[int], all 1s).
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    if user_prompt is None:
        user_prompt = "Write a brief caption for this video."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": caption},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoding = tokenizer(text, add_special_tokens=False, truncation=True, max_length=512)
    token_ids = encoding["input_ids"]
    loss_mask = [1] * len(token_ids)
    return {"token_ids": token_ids, "loss_mask": loss_mask}


def tokenize_sft(user_text: str, assistant_text: str, stage: int = 2, tokenizer=None) -> dict:
    """
    Stage 2/3 tokenization: answer-only loss on assistant tokens.

    Tokenizes the full chat sequence but sets loss_mask=0 for
    system+user tokens and loss_mask=1 for assistant tokens only.

    Parameters
    ----------
    user_text : str
        The user's question or prompt.
    assistant_text : str
        The assistant's response.
    stage : int
        Training stage (2 or 3). Currently unused but kept for API consistency.
    tokenizer : optional
        HuggingFace tokenizer. If None, uses cached default.

    Returns
    -------
    dict with token_ids (list[int]) and loss_mask (list[int]).
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoding = tokenizer(text, add_special_tokens=False, truncation=True, max_length=1024)
    token_ids = encoding["input_ids"]

    # Find where assistant response starts by tokenizing system+user only
    user_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    user_text_only = tokenizer.apply_chat_template(
        user_messages, tokenize=False, add_generation_prompt=True
    )
    user_encoding = tokenizer(user_text_only, add_special_tokens=False, truncation=True, max_length=1024)
    user_len = len(user_encoding["input_ids"])

    # 0 for system+user, 1 for assistant
    loss_mask = [0] * min(user_len, len(token_ids)) + [1] * max(0, len(token_ids) - user_len)
    return {"token_ids": token_ids, "loss_mask": loss_mask}
