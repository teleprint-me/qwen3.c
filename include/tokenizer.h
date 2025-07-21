/**
 * @file tokenizer.h
 * @brief Qwen3 Tokenizer API (Prompt Templates + Byte-Pair Encoding)
 *
 * This header provides the full interface for working with the Qwen3 tokenizer,
 * including prompt template handling, vocabulary lookup, and BPE tokenization.
 *
 * @note The implementation supports decoding, encoding, and BPE merging of
 * input text. Special tokens (e.g., "<|endoftext|>") are parsed during
 * encoding. The model uses canonical NFC normalization and greedy merging with
 * learned ranks.
 */

#ifndef QWEN_TOKENIZER_H
#define QWEN_TOKENIZER_H

#include <sys/types.h> // ssize_t

/**
 * @defgroup Constants Constants and Defaults
 * @{
 */

#define QTKN_MAGIC 0x71746B6E ///< "qtkn" file magic
#define QTKN_VERSION 2 ///< Tokenizer binary version
#define QTKN_VOCAB_SIZE 151936 ///< Default vocabulary size for Qwen3
#define QTKN_MAX_SEQ_LEN 32768 ///< Max sequence length supported

/**@}*/

/**
 * @defgroup TokenizerCore Tokenizer Core Structures
 * Vocabulary and BPE token mapping.
 * @{
 */

/**
 * @brief A vocabulary entry with merge rank.
 */
typedef struct TokenEntry {
    char* token; ///< Null-terminated UTF-8 token
    float score; ///< Merge rank score (higher is better)
} TokenEntry;

typedef struct TokenSpecial {
    // core ids
    int bos; // begin of seq (end of text)
    int eos; // end of seq (im end)
    int eot; // end of text (im start)
    int pad; // pad seq (same as bos)
    // think ids (aka reasoning)
    int bor; // begin of think (inclusion disables)
    int eor; // end of think
    // tool call ids
    int btc; // begin of tool call (begin tool call)
    int etc; // end of tool call (end tool call)
    // tool response ids
    int btr; // begin of tool response (tool response start)
    int etr; // end of tool response (tool response end)
} TokenSpecial;

/**
 * @brief Tokenizer state and vocabulary.
 */
typedef struct Tokenizer {
    TokenEntry* entries; ///< Vocabulary table (id → token)
    TokenSpecial special;
    int magic;
    int version;
    int vocab_size; ///< Total number of tokens
    int max_len; ///< Maximum UTF-8 length of any token
} Tokenizer;

/**
 * @brief Load a tokenizer from a binary model file.
 *
 * @param prefix File path prefix (expects `.tokenizer` file)
 * @return Tokenizer* tokenizer object
 */
Tokenizer* tokenizer_create(const char* prefix);

/**
 * @brief Free tokenizer resources.
 *
 * @param tokenizer Tokenizer object
 */
void tokenizer_free(Tokenizer* t);

/**@}*/

/**
 * @defgroup TokenMapping Token Mapping Functions
 * Vocabulary lookup functions.
 * @{
 */

/**
 * @brief Convert a token id to its UTF-8 string.
 *
 * @param t Tokenizer object
 * @param id Vocabulary index
 * @return char* UTF-8 string (owned by tokenizer)
 *
 * @note This is an O(1) operation.
 */
char* tokenizer_id_to_token(Tokenizer* t, int id);

/**
 * @brief Convert a UTF-8 token string to its id.
 *
 * @param t Tokenizer object
 * @param token UTF-8 string
 * @return int token id, or -1 if not found
 *
 * @note This is a linear scan, worst-case O(V) time.
 *       Consider replacing with a hash map for faster lookup.
 */
int tokenizer_token_to_id(Tokenizer* t, const char* token);

/**@}*/

/**
 * @defgroup TokenizerEncoder Tokenizer Encoder API
 * Full text-to-token encoder with BPE merge logic.
 * @{
 */

/**
 * @brief Encode a UTF-8 string into a sequence of token ids using greedy BPE.
 *
 * @param t Tokenizer object
 * @param text Null-terminated UTF-8 string
 * @param ids Output array of token ids (must be preallocated)
 * @param n_ids Pointer to integer to store output count
 *
 * This function performs:
 * - Initial UTF-8 parsing and special token detection (O(N) time, O(N) space)
 * - Greedy pairwise BPE merges based on learned scores (O(T² * V) worst-case)
 *
 * @note Space complexity is O(N + T), where T is the number of tokens before
 * merging.
 * @note Time complexity can be reduced with a merge cache or trie-based lookup.
 */
void tokenizer_encode(Tokenizer* t, char* text, int* ids, int* n_ids);

/**@}*/

#endif // QWEN_TOKENIZER_H
