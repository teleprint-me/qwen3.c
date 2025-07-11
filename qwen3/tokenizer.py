"""
@file qwen3.tokenizer

Qwen3 Tokenizer Exporter
=========================

This module extracts and converts a HuggingFace tokenizer (specifically Qwen3)
into a minimal binary format suitable for high-performance inference in C.

Key Components:
---------------
- `template_write(...)`: Renders chat templates using the tokenizer's Jinja2 format.
- `tokenizer_vocab(...)`: Parses vocabulary, merge ranks, and scoring data from tokenizer files.
- `tokenizer_write(...)`: Serializes token data to a compact binary file, including:
    - max token length
    - BOS/EOS token IDs
    - token byte encodings
    - merge-based pseudo-scores

Design Notes:
-------------
- Output format is custom and aligned for direct use in low-level inference engines.
- Token byte encoding follows GPT-2's byte-to-unicode reversible mapping.
- Template rendering inverts the `enable_thinking` logic due to model-specific behavior.
- No fallback or default token encoding is permitted: behavior must be deterministic.
- BOS/EOS IDs are extracted from `config.json` — fallback values are used if missing.

Binary File Layout:
-------------------
[int32] magic (0x71746B6E)  # 'qtkn'
[int32] version             # format version (1)
[int32] max_token_length
[int32] bos_token_id
[int32] eos_token_id
[
  float32 score,
  int32 length,
  uint8[] token_bytes
] * num_tokens

This module is designed for clarity and reproducibility over generality.
No additional dependencies beyond Jinja2, tokenizers, and transformers.

Usage:
------
$ python -m qwen3.tokenizer <output_prefix> <input_dir>
"""

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Template
from tokenizers import Tokenizer

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
        print(f"[Template] Rendered:\n{text}")
        print(f"[Template] Wrote {len(text)} bytes to {path}.")


#
# UTF-8 Code Point Conversion
#


def bytes_to_unicode() -> dict[int, str]:
    """Generate a GPT-2 Byte to Unicode map."""
    base = list(range(ord("!"), ord("~") + 1))
    base += list(range(ord("¡"), ord("¬") + 1))
    base += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = base[:]
    offset = 0  # Track added bytes
    for char in range(256):
        if char not in base:
            base.append(char)
            codepoints.append(256 + offset)
            offset += 1  # Added a new byte
    return dict(zip(base, map(chr, codepoints)))


def unicode_to_bytes(token: str) -> bytes:
    """Convert a token to UTF-8 byte sequence."""
    # Invert keys and values
    token_to_id: dict[str, int] = {t: i for i, t in bytes_to_unicode().items()}
    # Map standard unicode bytes
    codepoints: list[int] = []
    for char in token:
        if char in token_to_id:
            codepoints.append(bytes([token_to_id[char]]))
        # else: # @warning Enabling this will have consequences down the pipeline.
        #     codepoints.append(char.encode("utf-8"))  # unmapped unicode
    # Merge converted byte sequence
    return b"".join(codepoints)


#
# HuggingFace Tokenizer Conversion
#


@dataclass
class Vocab:
    size: int
    max_token_length: int
    bos_id: int
    eos_id: int
    tokens: list[str]
    scores: dict[str, float]


def tokenizer_config(tokenizer: Tokenizer) -> dict[str, any]:
    print("[Tokenizer] Loading config metadata.")

    # Hack: HuggingFace doesn’t expose bos/eos IDs directly
    config_path = Path(tokenizer.name_or_path) / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "bos_token_id" not in config:
        print("[Tokenizer] Warning: Using fallback bos id.")
    else:
        print(f"[Tokenizer] bos_id={config['bos_token_id']}")

    if "eos_token_id" not in config:
        print("[Tokenizer] Warning: Using fallback eos id.")
    else:
        print(f"[Tokenizer] eos_id={config['eos_token_id']}")

    if "vocab_size" not in config:
        print("[Tokenizer] Warning: Using fallback vocab size.")
    else:
        print(f"[Tokenizer] vocab_size={config['vocab_size']}")

    return config


def tokenizer_rank_table(tokenizer: Tokenizer) -> dict[str, int]:
    """Construct a merge pair to rank index mapping."""
    print("[Tokenizer] Generating merge-rank table.")

    tokenizer_path = Path(tokenizer.name_or_path) / "tokenizer.json"
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)

    merges = tokenizer_json["model"]["merges"]

    rank_table = {}
    for i, merge in enumerate(merges):
        pair = tuple(merge if isinstance(merge, list) else merge.split())
        token = "".join(pair)
        rank_table[token] = i

    return rank_table


def tokenizer_rank_scores(
    rank_table: dict[str, int], tokens: list[str]
) -> dict[str, float]:
    """Assign pseudo-scores to tokens based on merge ranks."""
    print("[Tokenizer] Generating merge-rank scores.")

    scores = {}
    for token in tokens:
        rank = rank_table.get(token)
        if rank is not None:
            scores[token] = -math.log(rank + 1)
        else:
            scores[token] = -1e6  # Base vocab token
    return scores


def tokenizer_vocab(tokenizer: Tokenizer) -> Vocab:
    """Load vocab mappings and tokenizer metadata."""
    print("[Tokenizer] Generating model vocabulary.")

    id_to_token = {v: k for k, v in tokenizer.get_vocab().items()}
    tokens = [id_to_token[i] for i in sorted(id_to_token)]

    # Compute target size
    config = tokenizer_config(tokenizer)
    vocab_size = config.get("vocab_size", 151936)
    current_size = len(tokens)
    missing = vocab_size - current_size

    if missing > 0:
        print(f"[Tokenizer] Missing {missing} tokens. Adding padded tokens.")
        start_id = current_size
        for i in range(missing):
            tokens.append(f"<|pad_{start_id + i}|>")
        assert len(tokens) == vocab_size, "Vocab size must be equal to model embed dim"

    max_token_length = max(len(t) for t in tokens)
    rank_table = tokenizer_rank_table(tokenizer)
    rank_scores = tokenizer_rank_scores(rank_table, tokens)

    print(f"[Tokenizer] Vocab has {len(tokens)} tokens with max len of {max_token_length}")
    return Vocab(
        size=vocab_size,
        max_token_length=max_token_length,
        bos_id=config.get("bos_token_id", 151643),
        eos_id=config.get("eos_token_id", 151645),
        tokens=tokens,
        scores=rank_scores,
    )


def tokenizer_write(vocab: Vocab, output_file: str) -> None:
    print("[Tokenizer] Serializing model tokenizer.")
    with open(output_file + ".tokenizer", "wb") as out_f:
        # Binary header
        out_f.write(struct.pack("I", 0x71746B6E))  # (qtkn) 4 bytes
        out_f.write(struct.pack("i", 1))  # 4 bytes
        out_f.write(struct.pack("i", vocab.size))
        out_f.write(struct.pack("i", vocab.max_token_length))
        out_f.write(struct.pack("i", vocab.bos_id))
        out_f.write(struct.pack("i", vocab.eos_id))

        # Tokens will be cast to uint8_t on C-side
        for token in vocab.tokens:
            token_bytes = unicode_to_bytes(token)
            out_f.write(struct.pack("f", vocab.scores[token]))  # float32 score
            out_f.write(struct.pack("i", len(token_bytes)))  # int32 length
            out_f.write(token_bytes)  # UTF-8 bytes

    print(f"[Tokenizer] Wrote tokenizer model to {output_file}.tokenizer")


if __name__ == "__main__":
    from argparse import ArgumentParser
    from transformers import AutoTokenizer

    parser = ArgumentParser()
    parser.add_argument("output_file", type=str, help="The output file path.")
    parser.add_argument("input_dir", type=str, help="The input dir path.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.input_dir)
    template_write(tokenizer, args.output_file)

    vocab = tokenizer_vocab(tokenizer)
    tokenizer_write(vocab, args.output_file)
