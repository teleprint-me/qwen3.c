"""
@file qwen3.weights
@brief Converts Qwen3ForCausalLM model weights into a custom binary format.

This module provides a full pipeline for exporting Qwen3 model weights from
PyTorch (transformers) into a C-compatible binary file format.

The export process includes:
- Loading model hyperparameters from config.json
- Initializing a minimal Transformer representation (for serialization)
- Converting selected weights to symmetric int8 quantization (Q8_0)
- Preserving normalization and scale parameters in float32
- Writing a fixed 256-byte header followed by serialized tensors

All exported weights are grouped and quantized with a configurable group size
(default: 64), and written in an order compatible with a C-based inference engine.

The output file consists of:
1. A 256-byte header with architecture metadata
2. LayerNorm and attention scale weights in float32
3. Model weights quantized in grouped Q8_0 format (int8 + scale)

Use `model_write()` to generate the final .bin file.

@see qwen3.model for internal architecture structure
@see qwen3.tokenizer for tokenizer and template serialization
"""

import json
import os
import struct
from io import BufferedWriter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch import nn
from qwen3.model import ModelArgs, Transformer

#
# Load params
#


def model_params(input_dir: str) -> ModelArgs:
    print("[Weights] Loading model hyper parameters.")

    params = ModelArgs()

    config_path = os.path.join(input_dir, "config.json")
    with open(config_path, "r") as f:
        config_json = json.load(f)

    params.dim = config_json.get("hidden_size", 2048)
    params.n_layers = config_json.get("num_hidden_layers", 28)
    params.n_heads = config_json.get("num_attention_heads", 16)
    params.n_kv_heads = config_json.get("num_key_value_heads", 8)
    params.vocab_size = config_json.get("vocab_size", 151936)
    params.hidden_dim = config_json.get("intermediate_size", 6144)
    params.norm_eps = config_json.get("rms_norm_eps", 1e-06)
    params.max_seq_len = config_json.get("max_position_embeddings", 40960)
    params.head_dim = config_json.get("head_dim", params.dim // params.n_heads)

    print(params)
    return params


#
# Load and initialize weights
#


def model_load(params: ModelArgs, state: dict[str, Tensor]) -> Transformer:
    print("[Weights] Loading and initializing Qwen3ForCausalLM weights.")

    model = Transformer(params)
    model.tok_embeddings.weight = nn.Parameter(state["model.embed_tokens.weight"])
    model.norm.weight = nn.Parameter(state["model.norm.weight"])

    for layer in model.layers:
        i = layer.layer_id
        layer.attention_norm.weight = nn.Parameter(
            state[f"model.layers.{i}.input_layernorm.weight"]
        )
        layer.attention.wq.weight = nn.Parameter(
            state[f"model.layers.{i}.self_attn.q_proj.weight"]
        )
        layer.attention.wk.weight = nn.Parameter(
            state[f"model.layers.{i}.self_attn.k_proj.weight"]
        )
        layer.attention.wv.weight = nn.Parameter(
            state[f"model.layers.{i}.self_attn.v_proj.weight"]
        )
        layer.attention.wo.weight = nn.Parameter(
            state[f"model.layers.{i}.self_attn.o_proj.weight"]
        )
        layer.attention.lq.weight = nn.Parameter(
            state[f"model.layers.{i}.self_attn.q_norm.weight"]
        )
        layer.attention.lk.weight = nn.Parameter(
            state[f"model.layers.{i}.self_attn.k_norm.weight"]
        )
        layer.ffn_norm.weight = nn.Parameter(
            state[f"model.layers.{i}.post_attention_layernorm.weight"]
        )
        layer.feed_forward.w1.weight = nn.Parameter(
            state[f"model.layers.{i}.mlp.gate_proj.weight"]
        )
        layer.feed_forward.w2.weight = nn.Parameter(
            state[f"model.layers.{i}.mlp.down_proj.weight"]
        )
        layer.feed_forward.w3.weight = nn.Parameter(
            state[f"model.layers.{i}.mlp.up_proj.weight"]
        )

    model.output.weight = nn.Parameter(state["lm_head.weight"])
    model.eval()
    return model


#
# Quantize weights
#


@dataclass
class Q8Tensor:
    """Quantized tensor with symmetric Q8_0 representation."""

    quant: Tensor  # int8 tensor
    scale: Tensor  # fp32 scale vector (per group)
    error: float  # max group-wise quantization error


def quantize_q8_0(tensor: Tensor, group_size: int) -> Q8Tensor:
    """
    Quantize a tensor symmetrically to int8 in Q8_0 format ([-127, 127]).

    Args:
        tensor: A float or half tensor to be quantized.
        group_size: Number of values per quantization group.

    Returns:
        Q8Tensor: Quantized output, scale factors, and max error.
    """
    assert (
        tensor.numel() % group_size == 0
    ), f"Tensor size {tensor.numel()} not divisible by group size {group_size}"

    # Flatten into [num_groups, group_size]
    w = tensor.float().reshape(-1, group_size)

    # Compute scale per group
    wmax = torch.abs(w).max(dim=1).values
    scale = wmax / 127.0  # ensures full dynamic range

    # Quantize
    quant = torch.round(w / scale[:, None]).to(torch.int8)

    # Dequantize to check reconstruction error
    dequant = (quant.float() * scale[:, None]).reshape(-1)
    error = torch.abs(dequant - w.view(-1)).max().item()

    return Q8Tensor(quant=quant, scale=scale, error=error)


#
# Serialize weights
#


def serialize_fp32(buffer: BufferedWriter, w: Tensor) -> None:
    """writes one fp32 tensor to file that is open in wb mode"""
    d = w.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f"{len(d)}f", *d)
    buffer.write(b)


def serialize_int8(buffer: BufferedWriter, w: Tensor) -> None:
    """writes one int8 tensor to file that is open in wb mode"""
    d = w.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f"{len(d)}b", *d)
    buffer.write(b)


#
# PyTorch weight conversion
#


@dataclass
class ExportGroup:
    size: int
    model: Transformer
    weights: list[Tensor] = field(default_factory=list)
    shared_classifier: bool = False
    buffer: Optional[BufferedWriter] = None


def export_group_adjust_size(group: ExportGroup) -> int:
    while group.model.params.dim % group.size != 0:
        group.size //= 2
        print(
            f"[Weights] Warning: Reducing group size to {group.size} to fit hidden_dim."
        )
    return group.size


def export_group_weights(group: ExportGroup) -> list[Tensor]:
    """
    Return weights in the fixed serialization order expected by the C-side.
    Does not include the output projection (classifier).
    """
    print("[Weights] Exporting grouped weights.")

    return [
        group.model.tok_embeddings.weight,
        *[layer.attention.wq.weight for layer in group.model.layers],
        *[layer.attention.wk.weight for layer in group.model.layers],
        *[layer.attention.wv.weight for layer in group.model.layers],
        *[layer.attention.wo.weight for layer in group.model.layers],
        *[layer.feed_forward.w1.weight for layer in group.model.layers],
        *[layer.feed_forward.w2.weight for layer in group.model.layers],
        *[layer.feed_forward.w3.weight for layer in group.model.layers],
    ]


def export_group_weights_are_tied(group: ExportGroup) -> bool:
    """
    Appends the output projection weight if it's not tied to embeddings.
    Returns True if tied, False otherwise.
    """
    shared = torch.equal(group.model.tok_embeddings.weight, group.model.output.weight)
    if not shared:
        group.weights.append(group.model.output.weight)
    return shared


def export_group_validate_weights(group: ExportGroup) -> None:
    for i, w in enumerate(group.weights):
        assert w.numel() % group.size == 0, (
            f"weight {i} with shape {tuple(w.shape)} has {w.numel()} elements, "
            f"not divisible by group_size {group.size}"
        )


def export_group_write_header(group: ExportGroup) -> None:
    """
    Writes a 256-byte model header to the given binary buffer.

    Header structure (byte offsets):
      - [  0] uint32 magic number ("qwen" = 0x7177656E)
      - [  4] int32  version (currently 1)
      - [  8] int32  dim
      - [ 12] int32  hidden_dim
      - [ 16] int32  n_layers
      - [ 20] int32  n_heads
      - [ 24] int32  n_kv_heads
      - [ 28] int32  vocab_size
      - [ 32] int32  max_seq_len
      - [ 36] int32  head_dim
      - [ 40] int32  shared_classifier (1 if tied, else 0)
      - [ 44] int32  group_size
      - [ 48-255] padding (zeros)
    """
    print("[Weights] Exporting model header: magic=qwen, version=1.")

    group.buffer.write(struct.pack("I", 0x7177656E))  # magic: "qwen"
    group.buffer.write(struct.pack("i", 1))  # version

    p = group.model.params
    hidden_dim = group.model.layers[0].feed_forward.w1.weight.shape[0]
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads

    fields = (
        p.dim,
        hidden_dim,
        p.n_layers,
        p.n_heads,
        n_kv_heads,
        p.vocab_size,
        p.max_seq_len,
        p.head_dim,
        int(group.shared_classifier),
        group.size,
    )
    group.buffer.write(struct.pack("iiiiiiiiii", *fields))

    pad = 256 - group.buffer.tell()
    assert pad >= 0, f"Header overflow: wrote more than 256 bytes!"
    group.buffer.write(b"\0" * pad)


def export_group_write_fp32_norm_weights(group: ExportGroup) -> None:
    """Serialize attention and MLP RMSNorm parameters as fp32."""
    print("[Weights] [float32] Exporting normalized weights.")

    for layer in group.model.layers:  # attention norms
        serialize_fp32(group.buffer, layer.attention_norm.weight)
    for layer in group.model.layers:  # MLP norms
        serialize_fp32(group.buffer, layer.ffn_norm.weight)
    serialize_fp32(group.buffer, group.model.norm.weight)  # final pre-classifier norm


def export_group_write_fp32_attn_weights(group: ExportGroup) -> None:
    """Serialize lq/lk RMSNorm weights (Qwen3-specific) as fp32."""
    print("[Weights] [float32] Exporting attention weights.")

    for layer in group.model.layers:
        serialize_fp32(
            group.buffer,
            (
                layer.attention.lq.weight
                if layer.attention.lq.weight is not None
                else torch.ones(group.model.params.head_dim)
            ),
        )
    for layer in group.model.layers:
        serialize_fp32(
            group.buffer,
            (
                layer.attention.lk.weight
                if layer.attention.lk.weight is not None
                else torch.ones(group.model.params.head_dim)
            ),
        )


def export_group_write_q8_weights(group: ExportGroup) -> None:
    """
    Quantizes and serializes a list of weights to Q8_0 format.

    Args:
        buffer: Open binary file to write serialized output.
        weights: List of float32 weight tensors to be quantized.
        group_size: Group size for quantization (must divide tensor.numel()).
    """
    print("[Weights] [q8_0] Exporting model weights.")

    errors = []

    for i, tensor in enumerate(group.weights):
        q8 = quantize_q8_0(tensor, group.size)
        serialize_int8(group.buffer, q8.quant)
        serialize_fp32(group.buffer, q8.scale)
        errors.append((q8.error, tensor.shape))

        print(
            f"[{i+1:03d}/{len(group.weights)}] "
            f"quantized {tuple(tensor.shape)} "
            f"to Q8_0 with max error {q8.error:.8f}"
        )

    if errors:
        errors.sort(reverse=True)
        print(f"max quantization group error across all weights: {errors[0][0]:.8f}")


def model_write(model: Transformer, output_file: str, group_size: int = 64) -> None:
    """
    Export the model weights in Q8_0 into .bin file to be read from C.
    That is:
    - quantize all weights to symmetric int8, in range [-127, 127]
    - all other tensors (the rmsnorm params) are kept and exported in fp32
    - quantization is done in groups of group_size to reduce the effects of any outliers
    """
    group = ExportGroup(model=model, size=group_size)

    group.size = export_group_adjust_size(group)
    group.weights = export_group_weights(group)
    group.shared_classifier = export_group_weights_are_tied(group)
    export_group_validate_weights(group)

    group.buffer = open(output_file, "wb")
    export_group_write_header(group)
    export_group_write_fp32_norm_weights(group)
    export_group_write_fp32_attn_weights(group)
    export_group_write_q8_weights(group)

    group.buffer.close()
    print(f"Wrote model checkpoint to {output_file}")


if __name__ == "__main__":
    from argparse import ArgumentParser
    from transformers import AutoModelForCausalLM

    parser = ArgumentParser()
    parser.add_argument("output_file", type=str, help="The output file path.")
    parser.add_argument("input_dir", type=str, help="The input dir path.")
    parser.add_argument(
        "-g",
        "--group-size",
        type=int,
        default=64,
        help="Number of values per quantization group.",
    )
    args = parser.parse_args()

    state = AutoModelForCausalLM.from_pretrained(args.input_dir).state_dict()
    params = model_params(args.input_dir)
    model = model_load(params, state)
    model_write(model, args.output_file, args.group_size)
