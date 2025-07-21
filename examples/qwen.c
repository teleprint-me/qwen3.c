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

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <fcntl.h>
#include <omp.h>

#define MAX_SEQ_LEN 32768

/**
 * @section CLI Options
 */

typedef enum Thinking {
    THINKING_OFF,
    THINKING_ON,
} Thinking;

typedef struct Options {
    char* prompt;  // optional (used in completions)
    char* system_prompt;  // optional (used in chat completions)
    char* path;  // required (e.g., model.bin)
    char* mode;  // "completion" or "chat"
    unsigned long long seed;  // seed rng with time by default
    Thinking thinking;  // 1 enables thinking
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
        .seed = (unsigned long long) time(NULL),
        .thinking = THINKING_OFF,
        .temperature = 1.0f,
        .top_p = 0.9f,
        .seq_len = MAX_SEQ_LEN,
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
                    o->seed = (unsigned long long) seed;
                }
            } break;
            case 'c': {
                int seq_len = abs(atoi(arg));
                o->seq_len = (seq_len > MAX_SEQ_LEN) ? MAX_SEQ_LEN : seq_len;
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
                o->thinking = atoi(arg) ? THINKING_ON : THINKING_OFF;
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
 * @section Transformer Model
 * @{
 */

typedef struct Qwen {
    Model* model;
    Tokenizer* tokenizer;
    Sampler* sampler;
} Qwen;

Qwen* qwen_create(Options* o) {
    Qwen* q = calloc(1, sizeof(Qwen));
    if (!q) {
        return NULL;
    }

    // build the Tokenizer via the tokenizer .bin file
    q->tokenizer = tokenizer_create(o->path);
    if (!q->tokenizer) {
        goto tokenizer_failed;
    }

    // build the Transformer via the model .bin file
    q->model = model_create(o->path, o->seq_len);
    if (!q->model) {
        goto model_failed;
    }

    // build the Sampler
    q->sampler = sampler_create(
        q->model->params.vocab_size, o->temperature, o->top_p, o->seed
    );
    if (!q->sampler) {
        goto sampler_failed;
    }

    return q;

sampler_failed:
    model_free(q->model);
model_failed:
    tokenizer_free(q->tokenizer);
tokenizer_failed:
    free(q);
    return NULL;
}

void qwen_free(Qwen* q) {
    if (q) {
        if (q->model) {
            model_free(q->model);
        }
        if (q->tokenizer) {
            tokenizer_free(q->tokenizer);
        }
        if (q->sampler) {
            sampler_free(q->sampler);
        }
        free(q);
    }
}

/** @} */

/**
 * @section Completions
 * @{
 */

/**
 * @brief Autoregressive text completion.
 *
 * Generates a continuation for the given input `prompt` using
 * the transformer, tokenizer, and sampler provided.
 *
 * The prompt is first encoded into token IDs. The transformer
 * consumes the prompt tokens sequentially (teacher forcing),
 * after which sampling begins for each subsequent token until:
 *   - The model predicts BOS/EOS (sequence termination), or
 *   - The context length (seq_len) is reached.
 *
 * @param transformer Pointer to the Transformer model.
 * @param tokenizer   Pointer to the Tokenizer (handles encoding/decoding).
 * @param sampler     Pointer to the Sampler (controls temperature/top-p).
 * @param prompt      UTF-8 input string to complete.
 *
 * @note This function streams tokens directly to stdout.
 * @note Deterministic generation only occurs if temperature ~0
 *       and the RNG seed is fixed.
 */
void completion(Qwen* qwen, Options* opts) {
    fprintf(stderr, "[Completion]\n");

    // Validate prompt
    if (!opts->prompt || 0 == strlen(opts->prompt)) {
        fprintf(
            stderr, "[Completion] Error: Missing prompt. Use -i 'string'.\n"
        );
        exit(EXIT_FAILURE);
    }

    // Encode the prompt into token IDs
    int* ids = calloc(
        strlen(opts->prompt) + 1, sizeof(int)
    );  // +1 for null terminator
    if (!ids) {
        fprintf(
            stderr,
            "[Completion] Error: Failed to allocate memory for input ids.\n"
        );
        exit(EXIT_FAILURE);
    }

    int n_ids = 0;
    tokenizer_encode(qwen->tokenizer, opts->prompt, ids, &n_ids);
    if (n_ids < 1) {
        fprintf(stderr, "[Completion] Error: Failed to encode input prompt.\n");
        free(ids);
        exit(EXIT_FAILURE);
    }

    // Initialize autoregressive state
    int token = ids[0];  // first token from prompt
    int next = 0;  // next token to be generated

    for (int pos = 0; pos < qwen->model->params.seq_len; pos++) {
        // Forward pass: compute logits for this position
        float* logits = forward(qwen->model, token, pos);

        // Teacher forcing for prompt, sampling for generation
        if (pos + 1 < n_ids) {
            next = ids[pos + 1];  // still consuming prompt
        } else {
            next = sample(qwen->sampler, logits);  // now generating
        }

        // Decode and stream the current token
        printf("%s", tokenizer_id_to_token(qwen->tokenizer, token));
        fflush(stdout);

        // Stop if BOS/EOS encountered after prompt
        if (next == qwen->tokenizer->special.bos
            || next == qwen->tokenizer->special.eos) {
            break;
        }

        // Advance autoregressive state
        token = next;
    }

    printf("\n");
    free(ids);
}

/** @} */

/**
 * @section Chat Completions
 * @{
 */

typedef struct ChatContext {
    char* buffer;  // growing UTF-8 string (conversation so far)
    size_t capacity;  // max allowed (MAX_SEQ_LEN)
    size_t size;  // current size
} ChatContext;

ChatContext* chat_context_create(size_t max_seq_len) {
    ChatContext* ctx = calloc(1, sizeof(ChatContext));
    if (!ctx) {
        return NULL;
    }

    ctx->buffer = calloc(max_seq_len, 1);
    if (!ctx->buffer) {
        return NULL;
    }

    ctx->capacity = max_seq_len;
    ctx->size = 0;
    return ctx;
}

bool chat_context_reset(ChatContext* ctx) {
    if (!ctx) {
        return false;
    }

    memset(ctx->buffer, 0, ctx->capacity);
    ctx->size = 0;
    return true;
}

void chat_context_free(ChatContext* ctx) {
    if (!ctx) {
        return;
    }
    free(ctx->buffer);
    free(ctx);
}

static inline bool chat_append(ChatContext* c, const char* fmt, ...) {
    if (!c || !fmt) {
        return false;
    }

    va_list args;
    va_start(args, fmt);
    int written = vsnprintf(
        c->buffer + c->size, c->capacity - c->size, fmt, args
    );
    va_end(args);

    if (written < 0 || c->size + written >= c->capacity) {
        fprintf(stderr, "[Chat] Context overflow!\n");
        return false;
    }

    c->size += written;

#ifdef DEBUG_CHAT
    fprintf(
        stderr, "[Chat] buffer (%zu/%zu)\n%s", c->size, c->capacity, c->buffer
    );
#endif

    return true;
}

bool chat_append_system(ChatContext* c, Tokenizer* t, const char* content) {
    return chat_append(
        c,
        "%s%s\n%s%s\n",
        tokenizer_id_to_token(t, t->special.eot),
        "system",
        content,
        tokenizer_id_to_token(t, t->special.eos)
    );
}

bool chat_append_user(
    ChatContext* c, Tokenizer* t, Thinking think, const char* content
) {
    if (!chat_append(
            c,
            "%s%s\n%s%s\n%s%s\n",
            tokenizer_id_to_token(t, t->special.eot),
            "user",
            content,
            tokenizer_id_to_token(t, t->special.eos),
            tokenizer_id_to_token(t, t->special.eot),
            "assistant"
        )) {
        return false;
    }

    if (THINKING_OFF == think) {
        if (!chat_append(
                c,
                "%s\n\n%s\n",
                tokenizer_id_to_token(t, t->special.bor),
                tokenizer_id_to_token(t, t->special.eor)
            )) {
            return false;
        }
    }

    return true;
}

bool chat_append_assistant(ChatContext* c, Tokenizer* t, const char* content) {
    return chat_append(
        c, "%s%s\n", content, tokenizer_id_to_token(t, t->special.eos)
    );
}

void chat_input(const char* prompt, char* buffer, size_t bufsize) {
    printf("%s", prompt);
    if (fgets(buffer, bufsize, stdin)) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';  // strip newline
        }
    }
}

static inline uint64_t time_now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (uint64_t) ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

void chat_completion(Qwen* qwen, Options* opts) {
    fprintf(stderr, "[ChatCompletion]\n");

    Model* m = qwen->model;
    Tokenizer* t = qwen->tokenizer;
    Sampler* s = qwen->sampler;

    char prompt[MAX_SEQ_LEN];
    ChatContext* c = chat_context_create(opts->seq_len);

    int user_turn = 1;  // user starts
    int user_id = 0;  // user token id
    int n_ids = 0;  // number of token ids
    int* ids = (int*) malloc(MAX_SEQ_LEN * sizeof(int));  // array of token ids

    // start the main loop
    int current = 0;  // stores the current token to feed into the transformer
    int next = 0;  // will store the next token in the sequence
    int pos = 0;  // position in the sequence

    uint64_t cycles = 0;  // number of tokens generated
    uint64_t pp = 0;  // prompt processing
    uint64_t tg = 0;  // token generation

    while (1) {
        // if context window is exceeded, clear it
        if (pos >= m->params.seq_len) {
            user_turn = 1;
            pos = 0;
        }

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            chat_context_reset(c);

            // get the user prompt from stdin
            chat_input("\n> ", prompt, sizeof(prompt));
            // terminate if user enters a blank prompt
            if (!*prompt) {
                break;
            }

            // Optional system prompt (NULL or a valid string)
            if (pos == 0 && opts->system_prompt) {
                chat_append_system(c, t, opts->system_prompt);
            }

            chat_append_user(c, t, opts->thinking, prompt);

            // encode the rendered prompt into tokens
            pp = time_now_ms();
            tokenizer_encode(t, c->buffer, ids, &n_ids);
            pp = time_now_ms() - pp;
            tg = time_now_ms();

            user_id = 0;  // reset the user index
            user_turn = 0;  // enable generation
            cycles = 0;
        }

        current = (user_id < n_ids) ? ids[user_id++] : next;
        float* logits = forward(m, current, pos);
        next = sample(s, logits);
        pos++;
        cycles++;

        // assistant is responding
        if (user_id >= n_ids) {
            if (next == t->special.bos || next == t->special.eos) {
                printf("\n");
                user_turn = 1;

                tg = time_now_ms() - tg;
                fprintf(
                    stderr,
                    "\n[pp %lums] [tg %lums] [t/ms %.3ft]\n",
                    pp,
                    tg,
                    (double) tg / cycles
                );
            } else {
                printf("%s", tokenizer_id_to_token(t, next));
                fflush(stdout);
            }
        }
    }

    free(ids);
    chat_context_free(c);
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

    Qwen* qwen = qwen_create(&opts);
    if (!qwen) {
        fprintf(stderr, "[Error] Failed to initialize Qwen\n");
        return EXIT_FAILURE;
    }

    if (strcmp(opts.mode, "completion") == 0) {
        completion(qwen, &opts);
    } else if (strcmp(opts.mode, "chat") == 0) {
        chat_completion(qwen, &opts);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", opts.mode);
    }

    qwen_free(qwen);
    return EXIT_SUCCESS;
}

/** @} */
