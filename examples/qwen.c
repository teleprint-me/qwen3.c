/**
 * @file examples/qwen.c
 * @brief Inference for Qwen-3 Transformer model in pure C, int8 quantized
 * forward pass.
 *
 * Qwen3 has the following features:
 *   - Type: Causal Language Models
 *   - Training Stage: Pretraining & Post-training
 *   - Number of Parameters: 0.6B, 1.7B, and 4B
 *   - Number of Embedding Parameters: ~0.4B
 *   - Number of Layers: 0.6B/1.7B -> 28, 4B -> 36
 *   - Number of Attention Heads (GQA): 0.6B/1.7B -> 16 for Q, 4B -> 32 for Q,
 * always 8 for KV
 *   - Context Length: 32768 natively and 131072 tokens with YaRN.
 *
 * @ref https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f
 * @ref https://en.wikipedia.org/wiki/Ship_of_Theseus
 */

#include <tokenizer.h>
#include <model.h>
#include <forward.h>
#include <sampler.h>
#include <qwen.h>
#include <completion.h>

#include <sys/mman.h>
#include <sys/time.h>
#include <time.h>

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <omp.h>

/**
 * @section CLI Options
 */

typedef struct Options {
    char* prompt;  // optional (used in completions)
    char* system_prompt;  // optional (used in chat completions)
    char* path;  // required (e.g., model.bin)
    char* mode;  // "completion" or "chat"
    uint64_t seed;  // seed rng with time by default
    Think think;  // 1 enables thinking
    int seq_len;  // max context length
    float temperature;  // 0.0f = deterministic and 1.0f = creative
    float top_p;  // nucleus sampling, 1.0 = off
} Options;

// read-only: do not allocate to this! copy strings from it as needed.
Options options_init(void) {
    // set default parameters
    return (Options) {
        .prompt = NULL,
        .system_prompt = NULL,
        .path = NULL,
        .mode = "chat",
        .seed = (uint64_t) time(NULL),
        .think = THINK_ON,
        .temperature = 1.0f,
        .top_p = 0.9f,
        .seq_len = QTKN_MAX_SEQ_LEN,
    };
}

static void options_usage(void) {
    fprintf(stderr, "Usage:   runq <checkpoint> [options]\n");
    fprintf(stderr, "Example: runq Qwen3-4B.bin -r 1\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  top-p (nucleus sampling), default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(
        stderr, "  -c <int>    context window size, 0 (default) = max_seq_len\n"
    );
    fprintf(stderr, "  -m <string> mode: completion|chat, default: chat\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -y <string> system prompt (chat mode)\n");
    fprintf(stderr, "  -r <int>    reasoning mode: 0=off, 1=thinking\n");
}

int options_parse(Options* o, int argc, char** argv) {
    if (argc < 2) {
        options_usage();
        return 1;
    }

    o->path = argv[1];

    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc || argv[i][0] != '-' || strlen(argv[i]) != 2) {
            options_usage();
            return 1;
        }

        char flag = argv[i][1];
        char* arg = argv[i + 1];

        switch (flag) {
            case 't':
                o->temperature = atof(arg);
                break;
            case 'p':
                o->top_p = atof(arg);
                break;
            case 's': {
                int seed = abs(atoi(arg));
                if (seed) {
                    o->seed = (uint64_t) seed;
                }
            } break;
            case 'c': {
                int seq_len = abs(atoi(arg));
                o->seq_len = (seq_len > QTKN_MAX_SEQ_LEN) ? QTKN_MAX_SEQ_LEN : seq_len;
            } break;
            case 'i':
                o->prompt = arg;
                break;
            case 'y':
                o->system_prompt = arg;
                break;
            case 'm':
                o->mode = arg;
                break;
            case 'r':
                o->think = atoi(arg) ? THINK_ON : THINK_OFF;
                break;
            default:
                options_usage();
                return 1;
        }
    }

    return 0;
}

/** @} */

/**
 * @section main
 * @{
 */

int main(int argc, char* argv[]) {
    Options opts = options_init();
    if (0 != options_parse(&opts, argc, argv)) {
        return EXIT_FAILURE;
    }

    QwenConfig cfg = {
        .path = opts.path,
        .think = opts.think,
        .seed = opts.seed,
        .temperature = opts.temperature,
        .top_p = opts.top_p,
        .seq_len = opts.seq_len,
    };

    Qwen* qwen = qwen_create(&cfg);
    if (!qwen) {
        fprintf(stderr, "[Error] Failed to initialize Qwen\n");
        return EXIT_FAILURE;
    }

    if (strcmp(opts.mode, "completion") == 0) {
        completion(qwen, opts.prompt);
    } else if (strcmp(opts.mode, "chat") == 0) {
        chat_completion(qwen, opts.system_prompt);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", opts.mode);
    }

    qwen_free(qwen);
    return EXIT_SUCCESS;
}

/** @} */
