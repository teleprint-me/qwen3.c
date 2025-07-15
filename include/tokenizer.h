/**
 * @file tokenizer.h
 * @brief Qwen3 Tokenizer API (Prompt Templates + Byte-Pair Encoding)
 *
 * This header provides the full interface for working with the Qwen3 tokenizer,
 * including prompt template handling, vocabulary lookup, and BPE tokenization.
 *
 * @note The implementation supports decoding, encoding, and BPE merging of input text.
 *       Special tokens (e.g., "<|endoftext|>") are parsed during encoding.
 *       The model uses canonical NFC normalization and greedy merging with learned ranks.
 */

#ifndef QWEN_TOKENIZER_H
#define QWEN_TOKENIZER_H

#include <sys/types.h> // ssize_t

/** 
 * @defgroup Constants Constants and Defaults
 * @{
 */

#define QTKN_MAGIC 0x71746B6E ///< "qtkn" file magic
#define QTKN_VERSION 1 ///< Tokenizer binary version
#define QTKN_VOCAB_SIZE 151936 ///< Default vocabulary size for Qwen3
#define QTKN_MAX_SEQ_LEN 32768 ///< Max sequence length supported

/**@}*/

/**
 * @defgroup PromptTemplate Prompt Template API
 * Functions for loading and managing system/user templates.
 * @{
 */

/**
 * @brief Prompt template object (system or user)
 */
typedef struct Template {
    char* data; ///< UTF-8 prompt string (dynamically allocated)
    ssize_t size; ///< Size in bytes (excluding null terminator)
} Template;

/**
 * @brief Load a chat prompt template from disk.
 *
 * @param in_file Path prefix (model checkpoint file, without `.template`)
 * @param enable_system Whether to include the system prompt
 * @param enable_thinking Whether to enable thinking mode (Qwen-style)
 * @return Template* dynamically allocated prompt object
 */
Template* template_create(const char* in_file, int enable_system, int enable_thinking);

/**
 * @brief Free a prompt template.
 *
 * @param template Template object
 */
void template_free(Template* template);

/**@}*/

/**
 * @defgroup TokenizerCore Tokenizer Core Structures
 * Vocabulary and BPE token mapping.
 * @{
 */

/**
 * @brief A vocabulary entry with merge rank.
 */
typedef struct Token {
    char* entry; ///< Null-terminated UTF-8 token
    float score; ///< Merge rank score (higher is better)
} Token;

/**
 * @brief Tokenizer state and vocabulary.
 */
typedef struct Tokenizer {
    Token* tokens; ///< Vocabulary table (id → token)
    Template* prompt; ///< User prompt template
    Template* system; ///< System + user prompt template
    int bos_id; ///< Beginning-of-sequence token id
    int eos_id; ///< End-of-sequence token id
    int vocab_size; ///< Total number of tokens
    int max_token_length; ///< Maximum UTF-8 length of any token
} Tokenizer;

/**
 * @brief Load a tokenizer from a binary model file.
 *
 * @param in_file Path prefix (expects `.tokenizer` file)
 * @param vocab_size Number of tokens (usually QTKN_VOCAB_SIZE)
 * @param enable_thinking Enables or disables "thinking" template mode
 * @return Tokenizer* tokenizer object
 */
Tokenizer* tokenizer_create(const char* in_file, int enable_thinking);

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
 * @note Space complexity is O(N + T), where T is the number of tokens before merging.
 * @note Time complexity can be reduced with a merge cache or trie-based lookup.
 */
void tokenizer_encode(Tokenizer* t, char* text, int* ids, int* n_ids);

/**@}*/

#endif // QWEN_TOKENIZER_H
