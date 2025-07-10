/**
 * @file examples/tokenizer.c
 */

#include "tokenizer.h"

#include <stdint.h>
#include <string.h>
#include <stdio.h>

#define VOCAB_SIZE 151669

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "%s <tokenizer-model> <text>\n", *argv);
        return 1;
    }

    const char* model_path = argv[1];
    Tokenizer* tokenizer = tokenizer_create(model_path, VOCAB_SIZE, 0);

    for (int i = 2; i < argc; i++) {
        int id = tokenizer_token_to_id(tokenizer, argv[i]);
        printf("[Tokenizer] id=%d\n", id);
        char* token = tokenizer_id_to_token(tokenizer, id);
        printf("[Tokenizer] id=%d, token=%s\n", id, token);
    }

    tokenizer_free(tokenizer);
    return 0;
}
