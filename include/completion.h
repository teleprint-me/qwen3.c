/**
 * @file completion.h
 * @brief High-level text generation API for Qwen3 models.
 *
 * Provides two autoregressive generation modes:
 *  - **completion()**: single-turn text completion (e.g., code, prose)
 *  - **chat_completion()**: multi-turn conversational generation with
 *    optional system prompts and reasoning tokens (<think>).
 *
 * Both functions stream decoded UTF-8 tokens directly to stdout.
 */

#ifndef QWEN_COMPLETION_H
#define QWEN_COMPLETION_H

#include "qwen.h"

/**
 * @brief Single-turn text completion.
 *
 * The given UTF-8 `prompt` is encoded into token IDs and consumed
 * sequentially by the transformer (teacher forcing). Generation begins
 * after the prompt is exhausted and continues until:
 *  - The model predicts BOS/EOS, or
 *  - The configured maximum sequence length is reached.
 *
 * @param qwen   Initialized Qwen instance (must not be NULL).
 * @param prompt Null-terminated UTF-8 input string.
 *
 * @note This function streams tokens directly to stdout as they are generated.
 * @note Deterministic generation only occurs if temperature ≈ 0 and the
 *       random seed is fixed.
 */
void completion(Qwen* qwen, char* prompt);

/**
 * @brief Multi-turn conversational chat completion.
 *
 * Generates a conversational response for the current user input,
 * maintaining context within the model’s attention window.
 * Optionally includes system-level instructions and reasoning
 * tokens (<think>), controlled by `qwen->config->think`.
 *
 * @param qwen          Initialized Qwen instance.
 * @param system_prompt Optional system-level instructions (NULL for none).
 *
 * @note This function maintains no persistent state between calls.
 *       The caller is responsible for managing conversational history.
 */
void chat_completion(Qwen* qwen, char* system_prompt);

#endif  // QWEN_COMPLETION_H
