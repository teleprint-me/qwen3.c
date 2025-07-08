"""
@file qwen3.tokenizer
@brief Converts a Tokenizer to a custom binary format.
"""

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Template
from tokenizers import Tokenizer
from transformers import AutoTokenizer


#
# HuggingFace Tokenizer
#


def tokenizer_load(input_dir: str) -> Tokenizer:
    return AutoTokenizer.from_pretrained(input_dir)


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
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    merges: list
    scores: list
    ranks: dict


def tokenizer_vocab(tokenizer: Tokenizer) -> Vocab:
    # Get ID to token mapping
    vocab = Vocab()
    vocab.id_to_token = {v: k for k, v in tokenizer.get_vocab().items()}
    vocab.token_to_id = [vocab.id_to_token[i] for i in sorted(vocab.id_to_token)]

    tokenizer_path = Path(tokenizer.name_or_path) / "tokenizer.json"
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)
        merges = tokenizer_json["model"]["merges"]

    # Build merge rank table
    for i, merge in merges:
        rank = tuple(merge if isinstance(merge, list) else merge.split())
    merge_rank = {
        "".join(): i
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


def tokenizer_write(tokenizer: Tokenizer, output_file: str) -> None:
    vocab = tokenizer_vocab(tokenizer)
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


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("output_file", type=str, help="The output file path.")
    parser.add_argument("input_dir", type=str, help="The input dir path.")
    args = parser.parse_args()

    tokenizer = tokenizer_load(args.input_dir)
    template_write(tokenizer, args.output_file)
