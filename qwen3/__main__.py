"""
@file qwen3.__main__
@brief Converts a Qwen3ForCausalLM architecture to a custom binary format.
"""

import argparse
from qwen3.convert import load_model, build_tokenizer, build_prompts, model_export

if __name__ == "__main__":
    # Parse CLI parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file", type=str, help="The output file path.")
    parser.add_argument("input_dir", type=str, help="The input dir path.")
    args = parser.parse_args()

    # Convert Qwen3 from pytorch to Q8 binary file
    model = load_model(args.input_dir)
    tokenizer = build_tokenizer(model, args.input_dir, args.output_file)
    build_prompts(tokenizer, args.output_file)
    model_export(model, args.output_file)
