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

from tokenizers import Tokenizer

QTKN_MAGIC = 0x71746B6E
QTKN_VERSION = 2

#
# HuggingFace Tokenizer Conversion
#
# @templates Requires special ids to reference the proper special tags.
# @warning Special tokens are handled inversely due to model behavior.
# - If thinking is enabled, the special tokens for reasoning are **excluded**.
# - If thinking is disabled, the special tokens are **included**.
#


@dataclass
class SpecialTokens:
    # core ids
    bos: int = 151643  # begin of seq (end of text)
    eos: int = 151645  # end of seq (im end)
    eot: int = 151644  # end of text (im start)
    pad: int = 151643  # pad seq (same as bos)
    # think ids (aka reasoning)
    bor: int = 151667  # begin of think (inclusion disables)
    eor: int = 151668  # end of think
    # tool call ids
    btc: int = 151657  # begin of tool call (begin tool call)
    etc: int = 151658  # end of tool call (end tool call)
    # tool response ids
    btr: int = 151665  # begin of tool response (tool response start)
    etr: int = 151666  # end of tool response (tool response end)
    # @todo See if qwen3 is fill-in-the-middle (FIM) compatible.
    # these are omitted for now for simplicity


@dataclass
class Vocab:
    size: int
    max_token_length: int
    tokens: list[str]
    scores: dict[str, float]
    special: SpecialTokens


def tokenizer_special_tokens(tokenizer: Tokenizer) -> SpecialTokens:
    print("[Tokenizer] Reading special ids.")

    # Hack: HuggingFace doesn’t expose bos/eos IDs directly
    SPECIAL_TOKEN_MAP = {
        "<|endoftext|>": "bos",
        "<|im_end|>": "eos",
        "<|im_start|>": "eot",
        "<think>": "bor",
        "</think>": "eor",
        "<tool_call>": "btc",
        "</tool_call>": "etc",
        "<tool_response>": "btr",
        "</tool_response>": "etr",
    }

    special = SpecialTokens()
    for token_id, token in tokenizer.added_tokens_decoder.items():
        field = SPECIAL_TOKEN_MAP.get(token.content)
        if field:
            print(f"[Tokenizer] content={token.content}, field={field}, id={token_id}")
            setattr(special, field, token_id)
            if field == "bos":  # pad mirrors bos
                special.pad = token_id

    if special.bos == -1 or special.eos == -1 or special.eot == -1:
        raise ValueError("[Tokenizer] BOS, EOS, and EOT must be defined.")

    return special


def tokenizer_config(tokenizer: Tokenizer) -> dict[str, any]:
    print("[Tokenizer] Loading config metadata.")

    # Hack: HuggingFace doesn’t expose bos/eos IDs directly
    config_path = Path(tokenizer.name_or_path) / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

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
    special = tokenizer_special_tokens(tokenizer)

    print(
        f"[Tokenizer] Vocab has {len(tokens)} tokens with max of {max_token_length} bytes."
    )

    return Vocab(
        size=vocab_size,
        max_token_length=max_token_length,
        tokens=tokens,
        scores=rank_scores,
        special=special,
    )


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


def tokenizer_write(vocab: Vocab, output_file: str) -> None:
    print("[Tokenizer] Serializing model tokenizer.")
    with open(output_file + ".tokenizer", "wb") as file:
        # Binary header
        file.write(struct.pack("I", QTKN_MAGIC))  # (qtkn) 4 bytes
        file.write(struct.pack("i", QTKN_VERSION))  # 4 bytes
        file.write(struct.pack("i", vocab.size))
        file.write(struct.pack("i", vocab.max_token_length))
        file.write(
            struct.pack(
                "10i",
                vocab.special.bos,
                vocab.special.eos,
                vocab.special.eot,
                vocab.special.pad,
                vocab.special.bor,
                vocab.special.eor,
                vocab.special.btc,
                vocab.special.etc,
                vocab.special.btr,
                vocab.special.etr,
            )
        )

        # Tokens will be cast to uint8_t on C-side
        for token in vocab.tokens:
            token_bytes = unicode_to_bytes(token)
            file.write(struct.pack("f", vocab.scores[token]))  # float32 score
            file.write(struct.pack("i", len(token_bytes)))  # int32 length
            file.write(token_bytes)  # UTF-8 bytes

    print(f"[Tokenizer] Wrote tokenizer model to {output_file}.tokenizer")


def tokenizer_validate(vocab: Vocab, output_file: str) -> None:
    print("[Tokenizer] Validating tokenizer model.")

    with open(output_file + ".tokenizer", "rb") as f:
        magic, version, size, max_len = struct.unpack("I i i i", f.read(16))
        specials = struct.unpack("10i", f.read(40))

    # --- Core Header ---
    assert magic == 0x71746B6E, f"[Magic] Expected 0x71746B6E, got 0x{magic:x}"
    assert version == 2, f"[Version] Expected 2, got {version}"
    assert size == vocab.size, f"[Size] Expected {vocab.size}, got {size}"
    assert (
        max_len == vocab.max_token_length
    ), f"[MaxLen] Expected {vocab.max_token_length}, got {max_len}"

    print(f"[Validate] magic=0x{magic:x}, version={version}")
    print(f"[Validate] vocab_size={size}, max_token_length={max_len}")

    # --- Special Tokens ---
    expected_specials = [
        vocab.special.bos,
        vocab.special.eos,
        vocab.special.eot,
        vocab.special.pad,
        vocab.special.bor,
        vocab.special.eor,
        vocab.special.btc,
        vocab.special.etc,
        vocab.special.btr,
        vocab.special.etr,
    ]

    for i, (expected, actual) in enumerate(zip(expected_specials, specials)):
        assert (
            expected == actual
        ), f"[Specials] Index {i}: Expected {expected}, got {actual}"
        print(f"[Validate] specials[{i}]={actual}")

    print("[Tokenizer] Validation successful.")


if __name__ == "__main__":
    from argparse import ArgumentParser
    from transformers import AutoTokenizer

    parser = ArgumentParser()
    parser.add_argument("output_file", type=str, help="The output file path.")
    parser.add_argument("input_dir", type=str, help="The input dir path.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.input_dir)
    vocab = tokenizer_vocab(tokenizer)
    tokenizer_write(vocab, args.output_file)
    tokenizer_validate(vocab, args.output_file)
