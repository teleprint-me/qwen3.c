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
    max_token_length: int
    bos_id: int
    eos_id: int
    token_to_id: list[str]
    scores: dict[str, float]


def tokenizer_config_ids(tokenizer: Tokenizer) -> tuple[str, str]:
    print("[Tokenizer] Loading bos and eos ids.")

    # Hack: HuggingFace doesn’t expose bos/eos IDs directly
    config_path = Path(tokenizer.name_or_path) / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "bos_token_id" not in config or "eos_token_id" not in config:
        print("[Tokenizer] Warning: Using fallback BOS/EOS IDs.")

    bos_id = config.get("bos_token_id", 151643)
    eos_id = config.get("eos_token_id", 151645)
    return bos_id, eos_id

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

    vocab_raw = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab_raw.items()}
    token_to_id = [id_to_token[i] for i in sorted(id_to_token)]
    max_token_length = max(len(t) for t in token_to_id)
    bos_id, eos_id = tokenizer_config_ids(tokenizer)
    rank_table = tokenizer_rank_table(tokenizer)
    rank_scores = tokenizer_rank_scores(rank_table, token_to_id)

    return Vocab(
        max_token_length=max_token_length,
        bos_id=bos_id,
        eos_id=eos_id,
        token_to_id=token_to_id,
        scores=rank_scores,
    )


def tokenizer_write(tokenizer: Tokenizer, output_file: str) -> None:
    vocab = tokenizer_vocab(tokenizer)

    print("[Tokenizer] Serializing model tokenizer.")
    with open(output_file + ".tokenizer", "wb") as out_f:
        # Binary header
        out_f.write(struct.pack("<i", 0x71746B6E))  # (qtkn) 4 bytes
        out_f.write(struct.pack("<i", 1))  # 4 bytes
        out_f.write(struct.pack("<i", vocab.max_token_length))
        out_f.write(struct.pack("<i", vocab.bos_id))
        out_f.write(struct.pack("<i", vocab.eos_id))

        # Tokens will be cast to uint8_t on C-side
        for token in vocab.token_to_id:
            token_bytes = unicode_to_bytes(token)
            out_f.write(struct.pack("f", vocab.scores[token]))  # float32 score
            out_f.write(struct.pack("<i", len(token_bytes)))  # int32 length
            out_f.write(token_bytes)  # UTF-8 bytes

    print(f"[Tokenizer] Wrote tokenizer model to {output_file}.tokenizer")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("output_file", type=str, help="The output file path.")
    parser.add_argument("input_dir", type=str, help="The input dir path.")
    args = parser.parse_args()

    tokenizer = tokenizer_load(args.input_dir)
    template_write(tokenizer, args.output_file)
    tokenizer_write(tokenizer, args.output_file)
