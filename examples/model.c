/**
 * @file examples/model.c
 */

#include "model.h"
#include <stdio.h>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "%s <model-file-path>\n", argv[0]);
        return 1;
    }

    int override_seq_len = 0;
    const char* model_path = argv[1];
    Model* m = model_create(model_path, override_seq_len);
    if (!m) {
        fprintf(
            stderr,
            "[Model] Failed to load Qwen3 checkpoint from %s",
            model_path
        );
        return 1;
    }

    fprintf(stderr, "[Model] Successfully loaded Qwen3 model.\n");
    model_free(m);
    return 0;
}
