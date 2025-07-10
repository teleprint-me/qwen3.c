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
    if (!file_path) return NULL;

    memcpy(file_path, in_file, strlen(in_file));
    memcpy(file_path + strlen(in_file), suffix, strlen(suffix));
    file_path[len] = '\0';

    FILE* file = fopen(file_path, "rb");
    free(file_path); // cleanup here is safe
    if (!file) return NULL;

    fseek(file, 0, SEEK_END);
    ssize_t size = ftell(file);
    rewind(file);

    Template* template = calloc(1, sizeof(Template));
    if (!template) { fclose(file); return NULL; }

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
    if (!t) return;
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
    if (!file_path) return NULL;

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

        if (fread(&score, sizeof(float), 1, file) != 1) break;
        if (fread(&length, sizeof(int), 1, file) != 1) break;

        char* buffer = calloc(length + 1, 1);
        if (!buffer || fread(buffer, 1, length, file) != (size_t)length) {
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
    if (!t) return;
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

char* tokenizer_id_to_token(Tokenizer *t, int id) {
    return t->tokens[id].entry;
}

int tokenizer_token_to_id(Tokenizer* t, const char* token) {
    if (!t || !t->tokens || !token) return -1;

    // find a match for str in vocab, return its index or -1 if not found
    for (int i = 0; i < t->vocab_size; i++) {
        if (0 == strcmp(token, t->tokens[i].entry)) {
            return i;
        }
    }
    return -1;
}

/** @} */
