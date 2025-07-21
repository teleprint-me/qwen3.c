/**
 * @file src/tokenizer.c
 */

#include "tokenizer.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/**
 * @section Tokenizer: BPE (Byte-pair Encoding) NFC (Normalization Form
 * Canonical Composition)
 * @{
 */

Tokenizer* tokenizer_create(const char* in_file) {
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
        free(file_path);
        return NULL;
    }
    free(file_path);

    Tokenizer* t = calloc(1, sizeof(Tokenizer));
    if (!t) {
        fclose(file);
        return NULL;
    }

    // Read header
    fread(&t->magic, sizeof(uint32_t), 1, file);
    fread(&t->version, sizeof(int32_t), 1, file);

    if (QTKN_MAGIC != t->magic || QTKN_VERSION != t->version) {
        fprintf(stderr, "[Tokenizer] Invalid tokenizer format.\n");
        fclose(file);
        return NULL;
    }

    fread(&t->vocab_size, sizeof(int32_t), 1, file);
    fread(&t->max_len, sizeof(int32_t), 1, file);
    fread(&t->special, sizeof(TokenSpecial), 1, file);

    t->entries = calloc(t->vocab_size, sizeof(TokenEntry));
    if (!t->entries) {
        fclose(file);
        free(t);
        return NULL;
    }

    // Read each token entry
    for (int i = 0; i < t->vocab_size; i++) {
        float score;
        int length;

        if (fread(&score, sizeof(float), 1, file) != 1) {
            fprintf(stderr, "[Tokenizer] Score read error at index %d\n", i);
            fclose(file);
            for (int k = 0; k < i; k++) {
                free(t->entries[k].token);
            }
            free(t->entries);
            free(t);
            return NULL;
        }

        if (fread(&length, sizeof(int), 1, file) != 1) {
            fprintf(stderr, "[Tokenizer] Length read error at index %d\n", i);
            fclose(file);
            for (int k = 0; k < i; k++) {
                free(t->entries[k].token);
            }
            free(t->entries);
            free(t);
            return NULL;
        }

        char* buffer = calloc(length + 1, 1);
        if (!buffer || fread(buffer, 1, length, file) != (size_t) length) {
            fprintf(stderr, "[Tokenizer] Token read error at index %d\n", i);
            if (buffer) {
                free(buffer);
            }
            fclose(file);
            for (int k = 0; k < i; k++) {
                free(t->entries[k].token);
            }
            free(t->entries);
            free(t);
            return NULL;
        }

        buffer[length] = '\0';
        t->entries[i].score = score;
        t->entries[i].token = buffer;
    }

    fclose(file);

    fprintf(stderr, "[Tokenizer] magic=%x\n", t->magic);
    fprintf(stderr, "[Tokenizer] version=%d\n", t->version);
    fprintf(stderr, "[Tokenizer] vocab_size=%d\n", t->vocab_size);
    fprintf(stderr, "[Tokenizer] max_len=%d\n", t->max_len);
    fprintf(stderr, "[Tokenizer] bos=%d\n", t->special.bos);
    fprintf(stderr, "[Tokenizer] eos=%d\n", t->special.eos);
    fprintf(stderr, "[Tokenizer] eot=%d\n", t->special.eot);

    return t;
}

void tokenizer_free(Tokenizer* t) {
    if (!t) {
        return;
    }
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->entries[i].token);
    }
    free(t->entries);
    free(t);
}

/** @} */

/**
 * @section Tokenizer: Token Mapping
 * @{
 */

char* tokenizer_id_to_token(Tokenizer* t, int id) {
    if (!t || !t->entries || id < 0 || id >= t->vocab_size) {
        fprintf(stderr, "[Tokenizer] ERROR: Invalid id! %d\n", id);
        return NULL;
    }
    return t->entries[id].token;
}

int tokenizer_token_to_id(Tokenizer* t, const char* token) {
    if (!t || !t->entries || !token) {
        fprintf(stderr, "[Tokenizer] ERROR: Invalid token! %s\n", token);
        return -1;
    }

    // find a match for str in vocab, return its index or -1 if not found
    for (int i = 0; i < t->vocab_size; i++) {
        if (!t->entries[i].token) {
            fprintf(
                stderr,
                "[Tokenizer] Error: Malformed entry! "
                "i=%d, token=%s, score=%f\n",
                (int) i,
                (char*) t->entries[i].token,
                (double) t->entries[i].score
            );
            return -1;
        }

        if (0 == strcmp(token, t->entries[i].token)) {
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
    for (int k = 0; start[k] && k < t->max_len; ++k) {
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
        char token[t->max_len];
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
                "[Tokenizer] Warning: Unknown character `%c` (codepoint %d)\n",
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
            t->max_len * 2 + 1,
            "%s%s",
            t->entries[ids[i]].token,
            t->entries[ids[i + 1]].token
        );

        int merged_id = tokenizer_token_to_id(t, buf);
        if (merged_id != -1 && t->entries[merged_id].score > best_score) {
            best_score = t->entries[merged_id].score;
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
    char* buffer = malloc(t->max_len * 2 + 1);
    while (1) {
        int best_id, best_idx;
        if (!tokenizer_find_best_merge(
                t, ids, *n_ids, buffer, &best_id, &best_idx
            )) {
            break;
        }

        ids[best_idx] = best_id;
        memmove(
            &ids[best_idx + 1],
            &ids[best_idx + 2],
            (*n_ids - best_idx - 2) * sizeof(int)
        );
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
