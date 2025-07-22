/**
 * @file src/completion.c
 */

#include "qwen.h"
#include "tokenizer.h"
#include "forward.h"
#include "completion.h"

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/**
 * @section Completions
 * @{
 */

void completion(Qwen* q, char* prompt) {
    fprintf(stderr, "[Completion]\n");

    if (!q || !q->model || !q->tokenizer || !q->sampler || !q->config) {
        fprintf(stderr, "[Completion] Error: Qwen is uninitialized.\n");
        exit(EXIT_FAILURE);
    }

    // Validate prompt
    if (!prompt || 0 == strlen(prompt)) {
        fprintf(stderr, "[Completion] Error: Missing prompt.\n");
        exit(EXIT_FAILURE);
    }

    // Encode the prompt into token IDs (+1 for null terminator)
    int* ids = calloc(strlen(prompt) + 1, sizeof(int));
    if (!ids) {
        fprintf(stderr, "[Completion] Error: Failed to allocate target ids.\n");
        exit(EXIT_FAILURE);
    }

    int n_ids = 0;
    tokenizer_encode(q->tokenizer, prompt, ids, &n_ids);
    if (n_ids < 1) {
        fprintf(stderr, "[Completion] Error: Failed to encode prompt.\n");
        free(ids);
        exit(EXIT_FAILURE);
    }

    // Initialize autoregressive state
    int token = ids[0];  // first token from prompt
    int next = 0;  // next token to be generated

    for (int pos = 0; pos < q->config->seq_len; pos++) {
        // Forward pass: compute logits for this position
        float* logits = forward(q->model, token, pos);

        // Teacher forcing for prompt, sampling for generation
        if (pos + 1 < n_ids) {
            next = ids[pos + 1];  // still consuming prompt
        } else {
            next = sample(q->sampler, logits);  // now generating
        }

        // Decode and stream the current token
        printf("%s", tokenizer_id_to_token(q->tokenizer, token));
        fflush(stdout);

        // Stop if BOS/EOS encountered after prompt
        if (next == q->tokenizer->special.bos
            || next == q->tokenizer->special.eos) {
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
 * @section Chat: Prompt Lifecycle
 * @{
 */

// Scratch buffer for user input
typedef struct ChatPrompt {
    char* buffer;  // input UTF-8 string
    size_t capacity;  // max allowed (MAX_SEQ_LEN)
    size_t size;  // current size
} ChatPrompt;

ChatPrompt* chat_prompt_create(size_t seq_len) {
    ChatPrompt* pmt = calloc(1, sizeof(ChatPrompt));
    if (!pmt) {
        return NULL;
    }

    pmt->buffer = calloc(seq_len, 1);
    if (!pmt->buffer) {
        free(pmt);
        return NULL;
    }

    pmt->capacity = seq_len;
    pmt->size = 0;
    return pmt;
}

bool chat_prompt_clear(ChatPrompt* pmt) {
    if (!pmt || !pmt->buffer) {
        return false;
    }

    memset(pmt->buffer, 0, pmt->capacity);
    pmt->size = 0;
    return true;
}

void chat_prompt_free(ChatPrompt* pmt) {
    if (pmt) {
        if (pmt->buffer) {
            free(pmt->buffer);
        }
        free(pmt);
    }
}

/** @} */

/**
 * @section Chat: Format user input
 */

static inline bool chat_format(ChatPrompt* pmt, const char* fmt, ...) {
    if (!pmt || !fmt) {
        return false;
    }

    va_list args;
    va_start(args, fmt);
    int written = vsnprintf(
        pmt->buffer + pmt->size, pmt->capacity - pmt->size, fmt, args
    );
    va_end(args);

    if (written < 0 || pmt->size + written >= pmt->capacity) {
        fprintf(stderr, "[Chat] Context overflow!\n");
        return false;
    }

    pmt->size += written;

#ifdef DEBUG_CHAT
    fprintf(
        stderr,
        "[Chat] buffer (%zu/%zu)\n%s",
        pmt->size,
        pmt->capacity,
        pmt->buffer
    );
#endif

    return true;
}

bool chat_format_system(ChatPrompt* pmt, Tokenizer* t, const char* content) {
    return chat_format(
        pmt,
        "%s%s\n%s%s\n",
        tokenizer_id_to_token(t, t->special.eot),
        "system",
        content,
        tokenizer_id_to_token(t, t->special.eos)
    );
}

bool chat_format_prompt(
    ChatPrompt* pmt, Tokenizer* t, Think think, const char* content
) {
    if (!chat_format(
            pmt,
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

    if (THINK_OFF == think) {
        if (!chat_format(
                pmt,
                "%s\n\n%s\n",
                tokenizer_id_to_token(t, t->special.bor),
                tokenizer_id_to_token(t, t->special.eor)
            )) {
            return false;
        }
    }

    return true;
}

/** @} */

/**
 * @section Chat Completions
 * @{
 */

typedef struct ChatState {
    // Flatten trivial user fields if you want less ceremony
    int* ids;
    int size;

    struct {
        int turn, id;
    } user;

    // Group only what makes sense
    struct {
        int id, next, pos;
    } model;

    struct {
        uint64_t cycles, pp, tg;
    } profile;
} ChatState;

static inline uint64_t time_now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (uint64_t) ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

void chat_buffer(const char* prompt, char* seq, int seq_len) {
    printf("%s", prompt);
    if (fgets(seq, seq_len, stdin)) {
        size_t len = strlen(seq);
        if (len > 0 && seq[len - 1] == '\n') {
            seq[len - 1] = '\0';  // strip newline
        }
    }
}

void chat_completion(Qwen* q, char* system_prompt) {
    fprintf(stderr, "[Chat]\n");

    if (!q || !q->model || !q->tokenizer || !q->sampler || !q->config) {
        fprintf(stderr, "[Chat] Error: Qwen is uninitialized.\n");
        exit(EXIT_FAILURE);
    }

    if (q->model->params.seq_len != q->config->seq_len) {
        fprintf(stderr, "[Chat] Error: Failed to sync seq_len.\n");
        exit(EXIT_FAILURE);
    }

    // Create prompt buffers
    char seq[q->config->seq_len];
    ChatPrompt* pmt = chat_prompt_create(q->config->seq_len);

    // Create the chat state
    ChatState state = {0};
    state.ids = (int*) malloc(q->config->seq_len * sizeof(int));
    state.user.turn = 1;

    while (1) {
        // if context window is exceeded, clear it
        if (state.model.pos >= q->config->seq_len) {
            state.user.turn = 1;
            state.model.pos = 0;
        }

        // when it is the user's turn to contribute tokens to the dialog...
        if (state.user.turn) {
            // Clear the input scratch buffer
            chat_prompt_clear(pmt);
            // get the user prompt from stdin
            chat_buffer("\n> ", seq, q->config->seq_len);

            // terminate if user enters a blank prompt
            if (!*seq) {
                break;
            }

            // Optional system prompt (NULL or a valid string)
            if (state.model.pos == 0 && system_prompt) {
                chat_format_system(pmt, q->tokenizer, system_prompt);
            }

            // Render templated prompt
            chat_format_prompt(pmt, q->tokenizer, q->config->think, seq);

            // Encode the rendered prompt into tokens
            state.profile.pp = time_now_ms();
            tokenizer_encode(q->tokenizer, pmt->buffer, state.ids, &state.size);
            state.profile.pp = time_now_ms() - state.profile.pp;

            state.user.id = 0;  // reset the user index
            state.user.turn = 0;  // enable generation

            state.profile.cycles = 0;  // reset token counter
            state.profile.tg = time_now_ms();
        }

        state.model.id = (state.user.id < state.size)
                             ? state.ids[state.user.id++]
                             : state.model.next;
        float* logits = forward(q->model, state.model.id, state.model.pos);
        state.model.next = sample(q->sampler, logits);
        state.model.pos++;
        state.profile.cycles++;

        // assistant is responding
        if (state.user.id >= state.size) {
            if (state.model.next == q->tokenizer->special.bos
                || state.model.next == q->tokenizer->special.eos) {
                printf("\n");
                state.user.turn = 1;

                state.profile.tg = time_now_ms() - state.profile.tg;
                fprintf(
                    stderr,
                    "\n[pp %lums] [tg %lums] [t/ms %.3ft]\n",
                    state.profile.pp,
                    state.profile.tg,
                    (double) state.profile.tg / state.profile.cycles
                );
            } else {
                char* token = tokenizer_id_to_token(
                    q->tokenizer, state.model.next
                );
                printf("%s", token);
                fflush(stdout);
            }
        }
    }

    free(state.ids);
    chat_prompt_free(pmt);
}

/** @} */
