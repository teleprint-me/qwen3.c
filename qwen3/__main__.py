"""
@file qwen3.__main__
@brief Converts a Qwen3ForCausalLM architecture to a custom binary format.
"""

from argparse import ArgumentParser, Namespace
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen3.weights import model_params, model_load, model_write
from qwen3.tokenizer import template_write, tokenizer_vocab, tokenizer_write


def parse_args() -> Namespace:
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Export the tokenizer model
    tokenizer = AutoTokenizer.from_pretrained(args.input_dir)
    template_write(tokenizer, args.output_file)
    vocab = tokenizer_vocab(tokenizer)
    tokenizer_write(vocab, args.output_file)

    # Export the model weights
    state = AutoModelForCausalLM.from_pretrained(args.input_dir).state_dict()
    params = model_params(args.input_dir)
    model = model_load(params, state)
    model_write(model, args.output_file, args.group_size)
