"""
@file qwen3.tokenizer
@brief Converts a Tokenizer to a custom binary format.
"""

from qwen3.model import Transformer
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from jinja2 import Template
import json
from pathlib import Path
import struct
import math
from dataclasses import dataclass

#
# Jinja2 Template Conversion
#
# @warning Special tokens are handled inversely due to model behavior.
# - If thinking is enabled, the special tokens for reasoning are **excluded**.
# - If thinking is disabled, the special tokens are **included**.
#

@dataclass
class TemplateConfig:
    suffix: str
    messages: list[dict[str, str]]
    enable_thinking: bool


# TODO: Add a psuedo-template for tool calling
def template_config() -> list[TemplateConfig]:
    print("[Template] Loading configuration.")
    base = [{"role": "user", "content": "%s"}]
    system = [{"role": "system", "content": "%s"}, {"role": "user", "content": "%s"}]
    return [
        TemplateConfig("", base, False),
        TemplateConfig(".with-thinking", base, True),
        TemplateConfig(".with-system", system, False),
        TemplateConfig(".with-system-and-thinking", system, True),
    ]

def template_render(tokenizer: Tokenizer) -> dict[str, str]:
    print("[Template] Rendering chat completion templates.")
    print(tokenizer.chat_template)
    template = Template(tokenizer.chat_template)
    return {
        cfg.suffix: template.render(
            messages=cfg.messages,
            add_generation_prompt=True,
            enable_thinking=cfg.enable_thinking,
        )
        for cfg in template_config()
    }

def template_write(tokenizer: Tokenizer, output_file: str) -> None:
    print("[Template] Writing to standard output.")
    rendered = template_render(tokenizer)
    for suffix, text in rendered.items():
        path = f"{output_file}.template{suffix}"
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        print(f"[Template] Wrote {path}")


#
# UTF-8 Unicode Conversion
#

def bytes_to_unicode() -> dict[int, str]:
    """Generate a GPT-2 Byte to Unicode map."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))

def unicode_to_bytes(token: str) -> bytes:
    """Convert a token to UTF-8 byte sequence"""
    # Invert keys and values
    utob = {u: b for b, u in bytes_to_unicode().items()}
    # Map characters to uint8_t (standard unicode byte)
    int8 = [bytes([utob[ch]]) if ch in utob else ch.encode("utf-8") for ch in token]
    # Merge converted byte sequence
    return b"".join(int8)


#
# HuggingFace Tokenizer Conversion
#


def tokenizer_write(model: Transformer, input_dir: str, output_file: str) -> Tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(input_dir)

    # Get ID to token mapping
    vocab = tokenizer.get_vocab()
    id_to_token: dict[int, str] = {v: k for k, v in vocab.items()}
    token_to_id: dict[str, int] = [id_to_token[i] for i in sorted(id_to_token)]

    tokenizer_path = Path(tokenizer.name_or_path)
    tokenizer_json_path = tokenizer_path / "tokenizer.json"

    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    # Extract vocab and merge rules
    vocab = tokenizer_data["model"]["vocab"]
    merges = tokenizer_data["model"]["merges"]

    # Build merge rank table
    merge_rank = {
        "".join(tuple(merge if isinstance(merge, list) else merge.split())): i
        for i, merge in enumerate(merges)
    }

    # Create pseudo-score dictionary
    # Tokens from initial vocab get score 0 (unmerged tokens)
    # Merged tokens get scores based on merge rank
    pseudo_scores = {}
    for _, token in enumerate(token_to_id):
        # If this token was the result of a merge, it will appear in merge_rank
        rank = merge_rank.get(token)

        if rank is not None:
            score = -math.log(rank + 1)
        else:
            score = -1e6  # Initial vocab tokens
        pseudo_scores[token] = score

    max_token_length = max(len(t) for t in token_to_id)

    # Write to binary
    with open(output_file + ".tokenizer", "wb") as out_f:
        # Header: max_token_length, bos_token_id, eos_token_id
        out_f.write(struct.pack("<I", max_token_length))
        out_f.write(struct.pack("<I", model.params.bos_id))
        out_f.write(struct.pack("<I", model.params.eos_id))

        for _, token in enumerate(token_to_id):
            token_bytes = unicode_to_bytes(token)
            out_f.write(struct.pack("f", pseudo_scores[token]))  # merge score
            out_f.write(struct.pack("<I", len(token_bytes)))  # 4 bytes: token length
            out_f.write(token_bytes)  # UTF-8 bytes

    print(f"Written tokenizer model to {output_file}.tokenizer")

    return tokenizer
