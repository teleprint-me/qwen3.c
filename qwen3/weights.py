"""
@file qwen3.weights
@brief Converts Qwen3ForCausalLM architectures to a custom binary format.

This module provides utilities for extracting and converting the weights from
float32 to symetric int8 quantization. The quantized weights are then serialized
into a custom binary format.
"""

import json
import os
import struct
from io import BufferedWriter
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch import nn
from transformers import AutoModelForCausalLM
from qwen3.model import ModelArgs, Transformer

#
# Load params
#


def model_params(input_dir: str) -> ModelArgs:
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


def write_q8_weights(buffer: BufferedWriter, weights: list[Tensor], group_size: int) -> None:
    """
    Quantizes and serializes a list of weights to Q8_0 format.

    Args:
        buffer: Open binary file to write serialized output.
        weights: List of float32 weight tensors to be quantized.
        group_size: Group size for quantization (must divide tensor.numel()).
    """
    errors = []

    for i, tensor in enumerate(weights):
        q8 = quantize_q8_0(tensor, group_size)
        serialize_int8(buffer, q8.quant)
        serialize_fp32(buffer, q8.scale)
        errors.append((q8.error, tensor.shape))

        print(
            f"{i+1}/{len(weights)} quantized {tuple(tensor.shape)} "
            f"to Q8_0 with max error {q8.error:.8f}"
        )

    if errors:
        errors.sort(reverse=True)
        print(f"max quantization group error across all weights: {errors[0][0]:.8f}")


#
# PyTorch weight conversion
#


def model_write(model: Transformer, output_file: str, group_size: int = 64) -> None:
    """
    Export the model weights in Q8_0 into .bin file to be read from C.
    That is:
    - quantize all weights to symmetric int8, in range [-127, 127]
    - all other tensors (the rmsnorm params) are kept and exported in fp32
    - quantization is done in groups of group_size to reduce the effects of any outliers
    """
    version = 1

    # let's first do some validation for this export type
    while model.params.dim % group_size != 0:
        group_size //= 2
        print(f"BACKOFF: reducing group size to {group_size} to fit hidden_dim")
    weights = [
        model.tok_embeddings.weight,
        *[layer.attention.wq.weight for layer in model.layers],
        *[layer.attention.wk.weight for layer in model.layers],
        *[layer.attention.wv.weight for layer in model.layers],
        *[layer.attention.wo.weight for layer in model.layers],
        *[layer.feed_forward.w1.weight for layer in model.layers],
        *[layer.feed_forward.w2.weight for layer in model.layers],
        *[layer.feed_forward.w3.weight for layer in model.layers],
    ]
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)

    if not shared_classifier:
        weights.append(model.output.weight)
    for w in weights:
        assert (
            w.numel() % group_size == 0
        ), f"weight {i} has numel {w.numel()}, not a multiple of group_size {group_size}"

    # write
    out_file = open(output_file, "wb")
    # first write out the header. the header will be 256 bytes
    # 1) write magic, which will be uint32 of "qwen" in ASCII
    out_file.write(struct.pack("I", 0x7177656E))
    # 2) write version, which will be int
    out_file.write(struct.pack("i", version))
    # 3) write the params, which will be 7 ints
    p = model.params
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack(
        "iiiiiiiiii",
        p.dim,
        hidden_dim,
        p.n_layers,
        p.n_heads,
        n_kv_heads,
        p.vocab_size,
        p.max_seq_len,
        p.head_dim,
        int(shared_classifier),
        group_size,
    )
    out_file.write(header)

    pad = 256 - out_file.tell()  # pad rest with zeros; tell returns current pos
    assert pad >= 0
    out_file.write(b"\0" * pad)
    # now that the header is done, let's write out the model

    # first let's write out all the params that we are keeping in fp32: the norms
    for layer in model.layers:  # attention norms
        serialize_fp32(out_file, layer.attention_norm.weight)
    for layer in model.layers:  # MLP norms
        serialize_fp32(out_file, layer.ffn_norm.weight)
    serialize_fp32(out_file, model.norm.weight)  # final pre-classifier norm

    # write out the QK-LayerNorm weights (Qwen3)
    for layer in model.layers:
        serialize_fp32(
            out_file,
            (
                layer.attention.lq.weight
                if layer.attention.lq.weight is not None
                else torch.ones(model.params.head_dim)
            ),
        )
    for layer in model.layers:
        serialize_fp32(
            out_file,
            (
                layer.attention.lk.weight
                if layer.attention.lk.weight is not None
                else torch.ones(model.params.head_dim)
            ),
        )

    write_q8_weights(out_file, weights, group_size)

    # write to binary file
    out_file.close()
    print(f"Written model checkpoint to {output_file}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("output_file", type=str, help="The output file path.")
    parser.add_argument("input_dir", type=str, help="The input dir path.")
    args = parser.parse_args()

    state = AutoModelForCausalLM.from_pretrained(args.input_dir).state_dict()

    params = model_params(args.input_dir)
    model = model_load(params, state)
