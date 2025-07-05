## qwen3.c

<p align="center">
  <img src="assets/qwen3_c.jpg" width="300" height="300" alt="Cute Llama">
</p>

**Run inference for frontier models based on the Qwen3 architecture, like Qwen3-4B or DeepSeek-R1-0528-Qwen3-8B, on your local Linux/macOS/Windows machine. No complicated configuration
required, just follow the steps below and enjoy.**

**Understand the basics of transformers but want to learn in-depth how LLM inference works? qwen3.c runs LLMs using one easy-to-understand (relatively speaking!) file of C source with no dependencies. Once you've
digested it and understand the data flow, you're there.**

This project's starting point was Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c), which does single-file
inference for LLaMA 2-compatible models. The LLaMA 2 architecture is now 2 years old (a lifetime in the field of AI) and is
long superseded. This project aims to maintain the simplicity of llama2.c while supporting a frontier
model architecture, with the goal of being both an up-to-date learning resource and also a great way to run the latest models locally.

Despite being only around 1000 lines of C code with no dependencies, qwen3.c supports everything you need to
enjoy running leading Qwen3-architecture LLMs on standard hardware (no GPUs needed), including multi-CPU core operation, support for Unicode/multi-language input and output, and thinking/reasoning models.

qwen3.c includes a Python tool to process any Qwen3-architecture HuggingFace model, converting to qwen3.c's model format which uses Q8_0 quantization for a good trade-off between quality
and performance.

## Clone the repo

```sh
git clone https://github.com/teleprint-me/qwen3.c qwen3
cd qwen3
```

## Setup the environment

### Create a virtual environment

```sh
python -m venv .venv
source .venv/bin/activate
```

### Install PyTorch

We only need the CPU version to convert the model in later steps.

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

_**NOTE:** This step must be completed **before** installing dependencies._

### Install dependencies

I removed the original versioning in the requirements.txt file so it's not bound.

```sh
pip install -r requirements.txt --upgrade 
```

For the frozen versioning, use `requirements.dev.txt` and omit the `--upgrade` flag.

```sh
pip install -r requirements.dev.txt
```

⚠️ **Do not install both** — it will cause dependency conflicts.

## Download the model

_**You can skip this step if you already have the original model files.**_

You'll need `huggingface-cli` to download the model. You'll need HuggingFace credentials for this step.

```sh
huggingface-cli login
```

Follow the prompts.

```sh
huggingface-cli whoami
```

Should output:

```sh
<user-name>
orgs: <org-name>
```

Create a local directory for the model files.

```sh
mkdir model
```

We need the physical directory with the actual files, so we need to specify the cache path.

```sh
huggingface-cli download Qwen/Qwen3-1.7B --cache-dir model
```

_Note that the 4B parameter model is preferred because it is more resiliant to quantization._
_The smaller models will suffer due to loss of precision which causes loss of coherency._
_This means that 1.7B is probably the smallest we can go without damaging the model._

It will take some time to download the model file. Time varies and will depend on your internet bandwidth.

## Convert the model

The model files are hashed (also just what it is), so we need to dig up the links to map the files.

```sh
model/models--Qwen--Qwen3-1.7B/snapshots/<UUID-DIR-PATH>
```

The export script loads the model file into memory using PyTorch and will take up a considerable amount of memory.
You will need a minimum of 32GB of CPU RAM to convert the model file. The 1.6B model may consume up to ~25.6GB of CPU RAM.

```sh
python -m qwen3 model/Qwen3-1.7B-Q8.bin model/models--Qwen--Qwen3-1.7B/snapshots/<UUID-DIR-PATH>
```

## Build the inference engine

The inference engine uses a single-threaded model for simplicity. This means it's very slow - even at Q8.
I plan on modding the implementation to support posix multi-threading for the matmuls later on.
For now, it is what it is.

```sh
make
```

To optimize the binary for faster inference, the [author of this repo](https://github.com/adriancable/qwen3.c)
recommends compiling with openmp.

```sh
make openmp
```

I just build with `gcc`. The original author set it so it uses openmp.

```sh
make gnu
```

The manual build command is just a one-liner that enables compiler optimizations.

```sh
gcc runq.c -o runq -lm -Ofast -fopenmp -march=native -D_FILE_OFFSET_BITS=64
```

## Inference the model

To view help, simply run the binary.

```sh
./runq
```

To run inference, we need to use a few flags from the help output.
The inference engine supports both completions and chat completions via a mode parameter.
For completions, use one of two keywords for the `-m` flag: `generate` or `chat` (default).

```sh
./runq model/Qwen3-1.7B-Q8.bin -m 'generate' -i 'Once upon a time'
```

Fun things you can try asking:

> Tell me a surprising fact about an animal of your choice.

> Write a short story for a 5 year old girl, featuring Sobieski the dog and Pepe the cat.

> Write a C program which sorts a list using the bubble sort algorithm.

> Write a poem about a little boy who builds a rocket to fly to the moon. In Japanese, please.

> Translate into English: 我希望您喜欢使用 qwen3.c 学习 LLM。

## Step 4: experiment with reasoning mode

qwen3.c also supports reasoning/thinking, if the model used supports it. Enable thinking with the `-r 1` command line parameter:

```aiignore
./runq Qwen3-4B.bin -r 1
```

Then try:

> Solve the quadratic equation x^2 - 5x + 6 = 0.

> What is 19673261 * 1842.64?

## Step 5: explore other models

Try for example DeepSeek-R1-0528-Qwen3-8B:

```aiignore
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
python export.py DeepSeek-R1-0528-Qwen3-8B.bin ./DeepSeek-R1-0528-Qwen3-8B
```

Then:

```aiignore
./runq DeepSeek-R1-0528-Qwen3-8B.bin
```

## Advanced options

qwen3.c lets you configure model settings via the command line including setting a system prompt, setting temperature, sampling parameters and so forth.
To show available settings, run qwen3.c without any command-line parameters:

```aiignore
./runq
```

## License

MIT License.

Original work by [Adrian Cable](https://github.com/adriancable/qwen3.c).
