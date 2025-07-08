"""
@file qwen3.convert
@brief Implements model conversion for Qwen3ForCausalLM architectures.
"""

import json
import math
import os
import struct
from pathlib import Path
from io import BufferedWriter

import numpy as np
import torch
from torch import Tensor
from jinja2 import Template
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenizers import Tokenizer
from qwen3.model import ModelArgs, Transformer

# -----------------------------------------------------------------------------
# common utilities


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


# TODO: Group size is the number of elements per thread
# NOTE: In model.py, this is equivalent to model_parallel_size
# This is not obvious at first glance and should be renamed for clarity.
def quantize_q80(w: Tensor, group_size: int) -> tuple[Tensor, Tensor, float]:
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float()  # convert to float32
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 127.0
    # scale into range [-127, 127]
    quant = w / scale[:, None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)
    # dequantize by rescaling
    fp32val = (int8val.float() * scale[:, None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # calculate the max error in each group
    err = torch.abs(fp32valr - w).max(dim=1).values
    # find the max error across all groups
    maxerr = err.max().item()
    return int8val, scale, maxerr


def model_export(model: Transformer, output_file: str, group_size: int = 64) -> None:
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

    # now let's write out all the params that we are quantizing to Q8_0
    # note we skip classifier weights, which are shared with the embedding
    ew = []
    for i, w in enumerate(weights):
        # quantize this weight
        q, s, err = quantize_q80(w, group_size)
        # save the int8 weights to file
        serialize_int8(out_file, q)  # save the tensor in int8
        serialize_fp32(out_file, s)  # save scale factors
        # logging
        ew.append((err, w.shape))
        print(
            f"{i+1}/{len(weights)} quantized {tuple(w.shape)} to Q8_0 with max error {err:.8f}"
        )

    # print the highest error across all weights, should be very small, e.g. O(~0.001)
    ew.sort(reverse=True)
    print(f"max quantization group error across all weights: {ew[0][0]:.8f}")

    # write to binary file
    out_file.close()
    print(f"Written model checkpoint to {output_file}")

#
# Load / import functions
#


def load_params(input_dir: str) -> ModelArgs:
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
    params.bos_id = config_json.get("bos_token_id", 151643)
    params.eos_id = config_json.get("eos_token_id", 151645)

    print(params)
    return params


def load_model(input_dir: str) -> Transformer:
    params = load_params(input_dir)
    model = Transformer(params)

    state_dict = AutoModelForCausalLM.from_pretrained(input_dir).state_dict()
    model.tok_embeddings.weight = nn.Parameter(state_dict["model.embed_tokens.weight"])
    model.norm.weight = nn.Parameter(state_dict["model.norm.weight"])
    for layer in model.layers:
        i = layer.layer_id
        layer.attention_norm.weight = nn.Parameter(
            state_dict[f"model.layers.{i}.input_layernorm.weight"]
        )
        layer.attention.wq.weight = nn.Parameter(
            state_dict[f"model.layers.{i}.self_attn.q_proj.weight"]
        )
        layer.attention.wk.weight = nn.Parameter(
            state_dict[f"model.layers.{i}.self_attn.k_proj.weight"]
        )
        layer.attention.wv.weight = nn.Parameter(
            state_dict[f"model.layers.{i}.self_attn.v_proj.weight"]
        )
        layer.attention.wo.weight = nn.Parameter(
            state_dict[f"model.layers.{i}.self_attn.o_proj.weight"]
        )
        layer.attention.lq.weight = nn.Parameter(
            state_dict[f"model.layers.{i}.self_attn.q_norm.weight"]
        )
        layer.attention.lk.weight = nn.Parameter(
            state_dict[f"model.layers.{i}.self_attn.k_norm.weight"]
        )
        layer.ffn_norm.weight = nn.Parameter(
            state_dict[f"model.layers.{i}.post_attention_layernorm.weight"]
        )
        layer.feed_forward.w1.weight = nn.Parameter(
            state_dict[f"model.layers.{i}.mlp.gate_proj.weight"]
        )
        layer.feed_forward.w2.weight = nn.Parameter(
            state_dict[f"model.layers.{i}.mlp.down_proj.weight"]
        )
        layer.feed_forward.w3.weight = nn.Parameter(
            state_dict[f"model.layers.{i}.mlp.up_proj.weight"]
        )

    # final classifier
    model.output.weight = nn.Parameter(state_dict["lm_head.weight"])
    model.eval()
    return model
