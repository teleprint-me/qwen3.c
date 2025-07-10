/**
 * @file include/tokenizer.h
 */

#ifndef QWEN_TOKENIZER_H
#define QWEN_TOKENIZER_H

#include <sys/types.h>

#define QTKN_MAGIC 0x71746B6E
#define QTKN_VERSION 1
#define QTKN_VOCAB_SIZE 151669
#define QTKN_MAX_SEQ_LEN 32768

/**
 * @section Tokenizer: Chat Template
 * @{
 */

typedef struct Template {
    char* data; // dynamically allocated UTF-8 string
    ssize_t size; // number of bytes read (excluding null terminator)
} Template;

Template* template_create(const char* in_file, int enable_system, int enable_thinking);
void template_free(Template* template);

/** @} */

/**
 * @section Tokenizer: BPE (Byte-pair Encoding) NFC (Normalization Form Canonical Composition)
 * @{
 */

typedef struct Token {
    char* entry; // null-terminated UTF-8 token
    float score; // merge rank or base token marker
} Token;

typedef struct Tokenizer {
    Token* tokens; // token → string + score
    Template* prompt; // user-only prompt
    Template* system; // system + user prompt
    int bos_id;
    int eos_id;
    int vocab_size;
    int max_token_length;
} Tokenizer;

Tokenizer* tokenizer_create(const char* in_file, int vocab_size, int enable_thinking);
void tokenizer_free(Tokenizer* tokenizer);

/** @} */

/**
 * @section Tokenizer: Token Mapping
 * @{
 */

// This is O(1) constant time and N input space.
char* tokenizer_id_to_token(Tokenizer* t, int id);

// This is O(V) to O(N log V) time and N input space.
// String comparison is the bottle-neck here.
int tokenizer_token_to_id(Tokenizer* t, const char* token);

/** @} */

/**
 * @section Tokenizer: Encoder
 * @{
 */

/**
 * | Stage            | Worst-Case Time      | Optimized Time (w/ hashmap) | Space Used         |
 * | ---------------- | -------------------- | --------------------------- | ------------------ |
 * | First-pass parse | `O(N * V)`           | `O(N * log V)` or `O(N)`    | `O(N)` for `ids[]` |
 * | BPE Merge Loop   | `O(T^2 * V)`         | `O(T^2)` or `O(T log T)`    | `O(T)`             |
 * | **Overall**      | `O(N * V + T^2 * V)` | `O(N + T log T)`            | `O(N + L)`         |
 *
 * Greedy BPE encoder, takes about O(N + T) in best case scenario.
 * The only way to speed this up is to use a hash table or a trie.
 * A hash-table and trie are omitted to retain some level of simplicity.
 */
void tokenizer_encode(Tokenizer* t, char* text, int* ids, int* n_ids);

/** @} */

#endif // QWEN_TOKENIZER_H
