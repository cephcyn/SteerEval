import os
from typing import Any, Optional, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

HF_TOKEN = os.environ.get("HF_TOKEN", "")


def reset_random(seed: int):
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def model_instruct_generate(
    tokenizer,
    model,
    messages,
    max_new_tokens: int = 5000,
    seed: Optional[int] = None,
) -> str:
    """
    Generate instruct model response using default chat template

    See also https://huggingface.co/docs/transformers/v5.0.0rc1/en/chat_templating#using-applychattemplate

    :param tokenizer: Model tokenizer object. Must be able to call apply_chat_template
    :param model: Some AutoModelForCausalLM-loadable object
    :param messages: Instruction format messages
    :param max_new_tokens: Maximum number of new tokens
    :type max_new_tokens: int
    :param seed: Random seed used for LLM prompting if not None
    :type seed: Optional[int]
    :return: Instruction model response
    :rtype: str
    """
    if seed is not None:
        reset_random(seed)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = ""
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id, # suppress excess logging
        )
    except Exception:
        print(prompt)
        print(inputs.input_ids.shape)

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1] :],
        skip_special_tokens=True,
    )
    return response


def load_model_pretrained_causallm(
    model_name: str,
    tokenizer_name: Optional[str] = None,
    model_cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    tokenizer_kwargs: dict[str, Any] = {},
    model_kwargs: dict[str, Any] = {},
) -> Tuple[Any, Any]:
    """
    Load a named pretrained AutoModelForCausalLM with associated tokenizer

    :param model_name: Model name
    :type model_name: str
    :param tokenizer_name: Tokenizer name. If None, uses model name.
    :type tokenizer_name: Optional[str]
    :param model_cache_dir: Directory to cache tok and model in
    :type model_cache_dir: Optional[str]
    :param token: HF access token (otherwise uses environment HF_TOKEN)
    :type token: Optional[str]
    :param tokenizer_kwargs: bonus kwargs for tokenizer load (see also: https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer.from_pretrained)
    :type tokenizer_kwargs: dict[str, Any]
    :param model_kwargs: bonus kwargs for model load (see also: https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained)
    :type model_kwargs: dict[str, Any]
    :return: Tuple of [tokenizer, model]
    :rtype: Tuple[Any, Any]
    """
    # check tokenizer name
    if tokenizer_name is None:
        tokenizer_name = model_name
    # override directory that we're stashing this in
    kwargs: dict[str, Any] = {}
    if model_cache_dir is not None:
        kwargs["cache_dir"] = model_cache_dir
    # hf access token?
    if token is None:
        token = HF_TOKEN
    if len(token) > 0:
        kwargs["token"] = token
    # load the tokenizer and model
    combined_kwargs = {}
    for k, v in kwargs.items():
        combined_kwargs[k] = v
    for k, v in tokenizer_kwargs.items():
        combined_kwargs[k] = v
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        **combined_kwargs,
    )
    # set pad token to eos token to suppress message
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # llm
    combined_kwargs = {}
    combined_kwargs["device_map"] = "auto"
    combined_kwargs["torch_dtype"] = torch.bfloat16
    for k, v in kwargs.items():
        combined_kwargs[k] = v
    for k, v in model_kwargs.items():
        combined_kwargs[k] = v
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **combined_kwargs,
    )
    return tokenizer, model
