/**
 * @file src/tokenizer.c
 */

#include "tokenizer.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/**
 * @section Tokenizer: Chat Template
 * @{
 */

Template* template_create(const char* in_file, int enable_system, int enable_thinking) {
    const char* suffix = NULL;
    if (enable_system) {
        suffix = enable_thinking ? ".template.with-system-and-thinking" : ".template.with-system";
    } else {
        suffix = enable_thinking ? ".template.with-thinking" : ".template";
    }

    // Construct full file path
    size_t len = strlen(in_file) + strlen(suffix);
    char* file_path = calloc(len + 1, 1);
    if (!file_path) {
        return NULL;
    }

    memcpy(file_path, in_file, strlen(in_file));
    memcpy(file_path + strlen(in_file), suffix, strlen(suffix));
    file_path[len] = '\0';

    FILE* file = fopen(file_path, "rb");
    free(file_path); // cleanup here is safe
    if (!file) {
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    ssize_t size = ftell(file);
    rewind(file);

    Template* template = calloc(1, sizeof(Template));
    if (!template) {
        fclose(file);
        return NULL;
    }

    template->size = size;
    template->data = calloc(size + 1, 1); // null-terminate for convenience
    if (!template->data) {
        fclose(file);
        free(template);
        return NULL;
    }

    fread(template->data, 1, size, file);
    fclose(file);
    return template;
}

void template_free(Template* t) {
    if (!t) {
        return;
    }
    free(t->data);
    free(t);
}

/** @} */

/**
 * @section Tokenizer: BPE (Byte-pair Encoding) NFC (Normalization Form Canonical Composition)
 * @{
 */

Tokenizer* tokenizer_create(const char* in_file, int vocab_size, int enable_thinking) {
    // Build file path for .tokenizer
    const char* suffix = ".tokenizer";
    size_t len = strlen(in_file) + strlen(suffix);
    char* file_path = calloc(len + 1, 1);
    if (!file_path) {
        return NULL;
    }

    memcpy(file_path, in_file, strlen(in_file));
    memcpy(file_path + strlen(in_file), suffix, strlen(suffix));

    FILE* file = fopen(file_path, "rb");
    if (!file) {
        fprintf(stderr, "[Tokenizer] Failed to open %s\n", file_path);
        return NULL;
    }
    free(file_path);

    // Read header
    uint32_t magic = 0;
    int32_t version = 0;
    fread(&magic, sizeof(uint32_t), 1, file);
    fread(&version, sizeof(int32_t), 1, file);

    if (QTKN_MAGIC != magic || QTKN_VERSION != version) {
        fprintf(stderr, "[Tokenizer] Invalid tokenizer format.\n");
        fclose(file);
        return NULL;
    }

    Tokenizer* t = calloc(1, sizeof(Tokenizer));
    if (!t) {
        fclose(file);
        return NULL;
    }

    t->vocab_size = vocab_size;

    fread(&t->max_token_length, sizeof(int32_t), 1, file);
    fread(&t->bos_id, sizeof(int32_t), 1, file);
    fread(&t->eos_id, sizeof(int32_t), 1, file);

    t->tokens = calloc(vocab_size, sizeof(Token));
    if (!t->tokens) {
        fclose(file);
        free(t);
        return NULL;
    }

    // Read each token entry
    for (int i = 0; i < vocab_size; i++) {
        float score;
        int length;

        if (fread(&score, sizeof(float), 1, file) != 1) {
            break;
        }
        if (fread(&length, sizeof(int), 1, file) != 1) {
            break;
        }

        char* buffer = calloc(length + 1, 1);
        if (!buffer || fread(buffer, 1, length, file) != (size_t) length) {
            fprintf(stderr, "[Tokenizer] Token read error at index %d\n", i);
            break;
        }
        buffer[length] = '\0';

        t->tokens[i].score = score;
        t->tokens[i].entry = buffer;
    }

    fclose(file);

    // Load prompt templates
    t->prompt = template_create(in_file, 0, enable_thinking);
    t->system = template_create(in_file, 1, enable_thinking);

    return t;
}

void tokenizer_free(Tokenizer* t) {
    if (!t) {
        return;
    }
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->tokens[i].entry);
    }
    free(t->tokens);
    template_free(t->prompt);
    template_free(t->system);
    free(t);
}

/** @} */

/**
 * @section Tokenizer: Token Mapping
 * @{
 */

char* tokenizer_id_to_token(Tokenizer* t, int id) {
    if (!t || id < 0 || id >= t->vocab_size) {
        return NULL;
    }
    return t->tokens[id].entry;
}

int tokenizer_token_to_id(Tokenizer* t, const char* token) {
    if (!t || !t->tokens || !token) {
        return -1;
    }

    // find a match for str in vocab, return its index or -1 if not found
    for (int i = 0; i < t->vocab_size; i++) {
        if (0 == strcmp(token, t->tokens[i].entry)) {
            return i;
        }
    }
    return -1;
}

/** @} */

/**
 * @section Tokenizer: Encoder
 * @{
 */

static int tokenizer_find_special_token(Tokenizer* t, char* start, char* out) {
    for (int k = 0; start[k] && k < t->max_token_length; ++k) {
        if (start[k] == '>') {
            int n_bytes = k + 1; // number of bytes consumed
            strncpy(out, start, n_bytes);
            out[n_bytes] = '\0';
            return n_bytes;
        }
    }
    return 0;
}

static int tokenizer_find_token_ids(Tokenizer* t, char* start, int* out) {
    int n_ids = 0;
    char* bytes = start;

    while (*bytes) {
        int id = -1;
        char token[t->max_token_length];
        token[0] = '\0';

        if (*bytes == '<') {
            int consumed = tokenizer_find_special_token(t, bytes, token);
            if (consumed > 0) {
                id = tokenizer_token_to_id(t, token);
                if (id != -1) {
                    bytes += consumed;
                }
            }
        }

        if (id == -1) {
            token[0] = *bytes++;
            token[1] = '\0';
            id = tokenizer_token_to_id(t, token);
        }

        if (id != -1) {
            out[(n_ids)++] = id;
        } else {
            fprintf(
                stderr,
                "Warning: Unknown character `%c` (codepoint %d)\n",
                token[0],
                token[0]
            );
        }
    }

    return n_ids;
}

static int tokenizer_find_best_merge(
    Tokenizer* t, int* ids, int n_ids, char* buf, int* out_id, int* out_index
) {
    float best_score = -1e10f;
    int best_id = -1;
    int best_idx = -1;

    for (int i = 0; i < n_ids - 1; ++i) {
        snprintf(
            buf,
            t->max_token_length * 2 + 1,
            "%s%s",
            t->tokens[ids[i]].entry,
            t->tokens[ids[i + 1]].entry
        );

        int merged_id = tokenizer_token_to_id(t, buf);
        if (merged_id != -1 && t->tokens[merged_id].score > best_score) {
            best_score = t->tokens[merged_id].score;
            best_id = merged_id;
            best_idx = i;
        }
    }

    if (best_idx != -1) {
        *out_id = best_id;
        *out_index = best_idx;
        return 1; // match found
    }

    return 0; // no match found
}

void tokenizer_merge_token_ids(Tokenizer* t, int* ids, int* n_ids) {
    char* buffer = malloc(t->max_token_length * 2 + 1);
    while (1) {
        int best_id, best_idx;
        if (!tokenizer_find_best_merge(t, ids, *n_ids, buffer, &best_id, &best_idx)) {
            break;
        }

        ids[best_idx] = best_id;
        memmove(&ids[best_idx + 1], &ids[best_idx + 2], (*n_ids - best_idx - 2) * sizeof(int));
        (*n_ids)--;
    }
    free(buffer);
}

void tokenizer_encode(Tokenizer* t, char* text, int* ids, int* n_ids) {
    // Initial tokenization
    *n_ids = tokenizer_find_token_ids(t, text, ids);
    tokenizer_merge_token_ids(t, ids, n_ids);
}

/** @} */
