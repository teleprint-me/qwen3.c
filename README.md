# qwen3.c

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

## Clone the Repository

```sh
git clone https://github.com/teleprint-me/qwen3.c qwen3
cd qwen3
```

## Setup the Environment

### 1. Create and Activate Virtual Environment

```sh
python -m venv .venv
source .venv/bin/activate
```

### 2. Install PyTorch (CPU-only)

> Must be installed **before** other dependencies.

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Python Dependencies

* For latest package versions:

  ```sh
  pip install -r requirements.txt --upgrade
  ```

* For pinned versions (reproducible builds):

  ```sh
  pip install -r requirements.dev.txt
  ```

> ⚠️ **Do not install both** — this may cause dependency conflicts.

## Download the Model

Skip this step if you already have the original HuggingFace model files.

### 1. Login to HuggingFace

```sh
pip install huggingface_hub
huggingface-cli login
```

Follow the prompts, then verify with:

```sh
huggingface-cli whoami
```

### 2. Download Model

```sh
mkdir model
huggingface-cli download Qwen/Qwen3-1.7B --cache-dir model
```

> ℹ️ The 4B model is **preferred** for quantization — smaller models (like 1.7B) may suffer significant degradation in coherency due to loss of precision.

## Convert the Model

The downloaded HuggingFace models are stored under a hashed directory path:

```
model/models--Qwen--Qwen3-1.7B/snapshots/<UUID>
```

You’ll need \~32GB RAM during this step. The export script loads the full model into memory.

```sh
python -m qwen3 model/Qwen3-1.7B-Q8.bin model/models--Qwen--Qwen3-1.7B/snapshots/<UUID>
```

## Build the Inference Engine

The inference engine is **single-threaded by default**.
OpenMP support can be enabled to utilize multiple CPU cores and speed up inference.
Note that even with Q8 quantization, performance is still limited.
Support for POSIX threads is planned for future versions.

### Build Options

#### Default (no OpenMP)

```sh
make run
```

#### With OpenMP

```sh
make openmp
```

#### Manual GCC Build (OpenMP + Optimizations)

```sh
make gnu
```

Or manually:

```sh
gcc runq.c -o runq -lm -Ofast -fopenmp -march=native -D_FILE_OFFSET_BITS=64
```

### Enable Multi-threading via OpenMP

By default, OpenMP uses **all threads** unless specified otherwise.

You can set the number of threads using the `OMP_NUM_THREADS` environment variable.

#### Option 1: Export Once

```sh
export OMP_NUM_THREADS=8
./runq model/Qwen3-1.7B-Q8.bin
```

> This sets it for the current shell session and all subsequent runs.

#### Option 2: Inline Per Run

```sh
OMP_NUM_THREADS=8 ./runq model/Qwen3-1.7B-Q8.bin
```

> This method is **one-shot** — it applies only to the current invocation.

For more details on using OpenMP with C, see the [OpenMP programming guide](https://curc.readthedocs.io/en/latest/programming/OpenMP-C.html).

## Run Inference

### View Help

```sh
./runq
```

### Basic Completion

```sh
./runq model/Qwen3-1.7B-Q8.bin -m generate -i 'Once upon a time'
```

> `-m` can be `generate` or `chat` (default).

#### Fun prompts to try:

* `Tell me a surprising fact about an animal of your choice.`
* `Write a short story for a 5-year-old featuring Sobieski the dog and Pepe the cat.`
* `Write a C program which sorts a list using the bubble sort algorithm.`
* `Translate into English: 我希望您喜欢使用 qwen3.c 学习 LLM。`

## Reasoning Mode

To enable math/thinking features (if supported by the model):

```sh
./runq model/Qwen3-4B-Q8.bin -r 1
```

Try:

* `Solve the quadratic equation x^2 - 5x + 6 = 0.`
* `What is 19673261 * 1842.64?`

## Try Other Models

Example: DeepSeek-R1-0528-Qwen3-8B

```sh
huggingface-cli download deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --cache-dir model

python -m qwen3 model/DeepSeek-R1-0528-Qwen3-8B-Q8.bin model/<model-dir>
./runq model/DeepSeek-R1-0528-Qwen3-8B-Q8.bin
```

## License

MIT License.

Original work by [Adrian Cable](https://github.com/adriancable/qwen3.c).
