/**
 * @file examples/tokenizer.c
 */

#include "tokenizer.h"

#include <stdint.h>
#include <string.h>
#include <stdio.h>

#define MAX_TOKENS 4096

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <tokenizer-model> <text>...\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    Tokenizer* tokenizer = tokenizer_create(model_path, QTKN_VOCAB_SIZE, 0);
    if (!tokenizer) {
        fprintf(stderr, "[Tokenizer] Failed to load tokenizer.\n");
        return 1;
    }

    for (int i = 2; i < argc; i++) {
        const char* text = argv[i];

        printf("Text: \"%s\"\n", text);

        // Encode input
        int ids[MAX_TOKENS];
        int n_ids = 0;
        encode(tokenizer, (char*) text, ids, &n_ids);

        // Print token IDs and corresponding text
        printf("Tokens: ");
        for (int j = 0; j < n_ids; j++) {
            printf("%d ", ids[j]);
        }
        printf("\n");

        printf("Decoded: ");
        for (int j = 0; j < n_ids; j++) {
            char* token = tokenizer_id_to_token(tokenizer, ids[j]);
            printf("'%s' ", token ? token : "<null>");
        }
        printf("\n\n");
    }

    tokenizer_free(tokenizer);
    return 0;
}
