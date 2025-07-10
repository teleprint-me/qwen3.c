#include "checkpoint.h"
#include <stdio.h>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "%s <model-file-path>\n", argv[0]);
        return 1;
    }

    int override_seq_len = 0;
    const char* model_path = argv[1];
    Transformer* transformer = transformer_create(model_path, override_seq_len);
    if (!transformer) {
        fprintf(stderr, "[Transformer] Failed to load model checkpoint from %s", model_path);
        return 1;
    }

    fprintf(stderr, "[Tranformer] Successfully loaded Qwen3 model.\n");
    transformer_free(transformer);
    return 0;
}
