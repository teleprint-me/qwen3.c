/**
 * @file runq.c
 * @brief Inference for Qwen-3 Transformer model in pure C, int8 quantized forward pass.
 *
 * Qwen3 has the following features:
 *   - Type: Causal Language Models
 *   - Training Stage: Pretraining & Post-training
 *   - Number of Parameters: 0.6B, 1.7B, and 4B
 *   - Number of Embedding Parameters: ~0.4B
 *   - Number of Layers: 0.6B/1.7B -> 28, 4B -> 36
 *   - Number of Attention Heads (GQA): 0.6B/1.7B -> 16 for Q, 4B -> 32 for Q, always 8 for KV
 *   - Context Length: 32768 natively and 131072 tokens with YaRN.
 * 
 * @ref https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <omp.h>

#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

// ----------------------------------------------------------------------------
// Globals
int GS = 1; // group size global for quantization of the weights

// ----------------------------------------------------------------------------
// Transformer model

typedef struct Config {
    int magic_number; // checkpoint magic number
    int version; // file format version
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
    int head_dim; // head dimension
    int shared_classifier; // 1 if wcls == p_tokens
    int group_size; // quantization group size (export.py uses 64)
} Config;

typedef struct QuantizedTensor {
    int8_t *q;    // quantized values
    float *s; // scaling factors
} QuantizedTensor;

/// @note Many of these are arrays of pointers, one per layer.
///       So memory layout looks like:
///           weights.wq[layer] → QuantizedTensor {q, s}
typedef struct TransformerWeights {
    // token embedding table
    QuantizedTensor *q_tokens; // (vocab_size, dim)
    float *token_embedding_table; // same, but dequantized
    // weights for rmsnorms
    float *rms_att_weight; // (layer, dim) rmsnorm weights
    float *rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    QuantizedTensor *wq; // (layer, dim, n_heads * head_size)
    QuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wo; // (layer, n_heads * head_size, dim)
    // QK-RMSNorm for Qwen3
    float *q_ln_weights;
    float *k_ln_weights;
    // weights for ffn (gate, down, up)
    QuantizedTensor *w1; // (layer, hidden_dim, dim)
    QuantizedTensor *w2; // (layer, dim, hidden_dim)
    QuantizedTensor *w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float *rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    QuantizedTensor *wcls;
} TransformerWeights;

typedef struct RunState {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float *key_cache;   // (layer, seq_len, dim)
    float *value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct Transformer {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    float *data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState* s, Config *p) {
    // we calloc instead of malloc to keep valgrind happy
    int all_heads_dim = p->n_heads * p->head_dim;
    int kv_dim = p->n_kv_heads * p->head_dim;

    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(all_heads_dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->xq = (QuantizedTensor) { .q = calloc(all_heads_dim, sizeof(int8_t)), .s = calloc(all_heads_dim / GS, sizeof(float)) };
    s->hq = (QuantizedTensor) { .q = calloc(p->hidden_dim, sizeof(int8_t)), .s = calloc(p->hidden_dim / GS, sizeof(float)) };
    s->q = calloc(all_heads_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * (uint64_t)p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * (uint64_t)p->seq_len * kv_dim, sizeof(float));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->att || !s->logits || !s->key_cache
     || !s->value_cache) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->xq.q);
    free(s->xq.s);
    free(s->hq.q);
    free(s->hq.s);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

// ----------------------------------------------------------------------------
// Quantization functions

void dequantize(QuantizedTensor *qx, float *x, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

void quantize(QuantizedTensor *qx, float *x, int n) {
    const int num_groups = n / GS;
    const float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++) {
        float* xg  = x + group * GS;
        int8_t* qg = qx->q + group * GS;

        // Find max absolute value
        float wmax = fabsf(xg[0]);
        #pragma omp simd reduction(max:wmax)
        for (int i = 1; i < GS; i++) {
            wmax = fmaxf(wmax, fabsf(xg[i]));
        }

        float scale = (wmax == 0.0f) ? 1e-6f : (wmax / Q_MAX); // avoid div by 0
        qx->s[group] = scale;

        /// @note Clamp to [-127, 127] to avoid int8 overflow on rare large values.
        #pragma omp parallel for
        for (int i = 0; i < GS; i++) {
            float q = xg[i] / scale;
            qg[i] = (int8_t) fminf(fmaxf(roundf(q), -Q_MAX), Q_MAX);
        }
    }
}

/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
    void *p = *ptr;
    QuantizedTensor *res = malloc(n * sizeof(QuantizedTensor));

    for (int i = 0; i < n; i++) {
        /* map quantized int8 values*/
        res[i].q = (int8_t*)p;
        p = (int8_t*)p + size_each;
        /* map scale factors */
        res[i].s = (float*)p;
        p = (float*)p + size_each / GS;
    }
    *ptr = p; // advance ptr to current position
    return res;
}

void memory_map_weights(TransformerWeights *w, Config *p, void *ptr) {
    // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights)
    float *fptr = (float*) ptr; // cast our pointer to float*

    w->rms_att_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr;
    fptr += p->dim;
    w->q_ln_weights = fptr;
    fptr += p->n_layers * p->head_dim;
    w->k_ln_weights = fptr;
    fptr += p->n_layers * p->head_dim;

    // now read all the quantized weights
    ptr = (void *)fptr; // now cast the pointer back to void*
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
    // dequantize token embedding table
    w->token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
    dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * p->head_dim));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * p->head_dim));
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * p->head_dim));
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * p->head_dim) * p->dim);

    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

    w->wcls = p->shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}

void read_checkpoint(char *checkpoint, Config *config, TransformerWeights* weights, float** data, ssize_t* file_size, int ctx_length) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open checkpoint %s\n", checkpoint); exit(EXIT_FAILURE); }

    #if defined _WIN32
        _fseeki64(file, 0, SEEK_END); // move file pointer to end of file
        *file_size = _ftelli64(file); // get the file size, in bytes
    #else
        fseek(file, 0, SEEK_END); // move file pointer to end of file
        *file_size = ftell(file); // get the file size, in bytes
    #endif

    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, fileno(file), 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    fclose(file);

    // checkpoint format is 256-byte header, and then the model weights

    memcpy(config, *data, sizeof(Config));
    if (config->magic_number != 0x7177656E) { fprintf(stderr, "File %s is not a qwen3.c checkpoint\n", checkpoint); exit(EXIT_FAILURE); }
    if (config->version != 1) { fprintf(stderr, "Checkpoint %s is version %d, need version 1\n", checkpoint, config->version); exit(EXIT_FAILURE); }

    if (ctx_length != 0 && ctx_length <= config->seq_len)
        config->seq_len = ctx_length;

    GS = config->group_size; // set as global, as it will be used in many places

    void *weights_ptr = ((char *)*data) + 256; // skip the header (256 bytes)
    memory_map_weights(weights, config, weights_ptr);
}

void build_transformer(Transformer *t, char *checkpoint_path, int ctx_length) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->data, &t->file_size, ctx_length);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer *t) {
    // free QuantizedTensors
    free(t->weights.q_tokens);
    free(t->weights.token_embedding_table);
    free(t->weights.wq);
    free(t->weights.wk);
    free(t->weights.wv);
    free(t->weights.wo);
    free(t->weights.w1);
    free(t->weights.w2);
    free(t->weights.w3);
    if(t->weights.wcls != t->weights.q_tokens) free(t->weights.wcls);
    // close the memory mapping
    if (t->data != MAP_FAILED) munmap(t->data, t->file_size);
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float *o, float *x, float *weight, int size) {
    // calculate sum of squares
    float sos = 0.0f;
    #pragma omp parallel for reduction(+:sos)
    for (int j = 0; j < size; j++) {
        sos += x[j] * x[j];
    }

    sos = 1.0f / sqrtf((sos / size) + 1e-6f);

    // normalize and scale
    #pragma omp parallel for
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (sos * x[j]);
    }
}

/// @todo Cache max_val if last token's softmax is reused (e.g. in KV cache).
/// @todo Replace expf() with fast_approx_exp() if precision tradeoff is acceptable.
/// @todo Consider log-sum-exp for log-domain ops.
void softmax(float *x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0]; // use first element as initial guess
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    // exponentiate and sum
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // normalize
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float *xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized

    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        float val = 0;
        int in = i * n;

        // do the matmul in groups of GS
        for (int j = 0; j <= n - GS; j += GS) {
            int32_t ival = 0;
            for (int k = 0; k < GS; k++)
                ival += x->q[j + k] * w->q[in + j + k];

            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
        }

        xout[i] = val;
    }
}

/// @todo Add precomputed rotary cache.
void rotary(float* t, int head_dim, int pos) {
    int half_dim = head_dim / 2;

    #pragma omp parallel for
    for (int i = 0; i < half_dim; i++) {
        float angle = pos * powf(1e6f, -(float)i / half_dim);
        float cos_a = cosf(angle), sin_a = sinf(angle);

        float real = t[i];
        float imag = t[i + half_dim];

        t[i]            = real * cos_a - imag * sin_a;
        t[i + half_dim] = real * sin_a + imag * cos_a;
    }
}

void attention(Config* p, RunState* s, int l, int pos) {
    int head_dim = p->head_dim;
    int kv_mul   = p->n_heads / p->n_kv_heads;
    int kv_dim   = p->n_kv_heads * head_dim;
    uint64_t loff = (uint64_t) l * p->seq_len * kv_dim;

    for (int h = 0; h < p->n_heads; h++) {
        float* q   = s->q + h * head_dim;
        float* att = s->att + h * p->seq_len;
        float* xb  = s->xb  + h * head_dim;

        // Compute attention scores
        #pragma omp parallel for
        for (int t = 0; t <= pos; t++) {
            float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
            float score = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                score += q[i] * k[i];
            }

            att[t] = score / sqrtf((float)head_dim);
        }

        // Normalize scores to get attention weights
        softmax(att, pos + 1);

        // Initialize output vector for this head
        memset(xb, 0, sizeof(float) * head_dim);

        // Accumulate weighted values in thread-safe way
        #pragma omp parallel
        {
            float tmp[head_dim];
            memset(tmp, 0, head_dim * sizeof(float));

            #pragma omp for
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
                float a = att[t];
                for (int i = 0; i < head_dim; i++) {
                    tmp[i] += a * v[i];
                }
            }

            // Reduce thread-local buffer into shared output
            for (int i = 0; i < head_dim; i++) {
                #pragma omp atomic
                xb[i] += tmp[i];
            }
        }
    }
}

/// @brief σ(x) = 1 / 1 + exp(-x)
/// @note This can not be modified. The model will become incoherent.
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/// @brief x∗σ(x), where σ(x) is the logistic sigmoid.
float silu(float x) {
    return x * sigmoid(x);
}

/// @brief SwiGLU(x) = silu(W₁x) ⊙ W₃x
void swish(RunState* s, int hidden_dim) {
    float* hb  = s->hb;  // W1(x)
    float* hb2 = s->hb2; // W3(x)

    #pragma omp parallel for
    for (int i = 0; i < hidden_dim; i++) {
        hb[i] = silu(hb[i]) * hb2[i];
    }
}

float *forward(Transformer *transformer, int token, int pos) {
    // a few convenience variables
    Config *p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = p->n_kv_heads * p->head_dim;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int all_heads_dim = p->n_heads * p->head_dim;
    // copy the token embedding into x
    memcpy(x, w->token_embedding_table + token*dim, dim * sizeof(float));

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {
        // KV cache layer offset
        uint64_t loff = l * (uint64_t)p->seq_len * kv_dim;

        // Save KV at this time step (pos) to cache
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // Normalize the input to attention.
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // Quantize for matmul efficiency.
        quantize(&s->xq, s->xb, dim);

        // Compute Q, K, V for this timestep.
        matmul(s->q, &s->xq, w->wq + l, dim, all_heads_dim);
        matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);

        float *gq = w->q_ln_weights + l * p->head_dim;   // 128 floats
        float *gk = w->k_ln_weights + l * p->head_dim;   // 128 floats

        /**
         * Q-RMSNorm + rotate each query head
         */
        for (int h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * p->head_dim;

            rmsnorm(q, q, gq, p->head_dim);
            rotary(q, p->head_dim, pos);
        }

        /**
         * K-RMSNorm + rotate each key head
         */
        for (int h = 0; h < p->n_kv_heads; h++) {
            float *k = s->k + h * p->head_dim;

            rmsnorm(k, k, gk, p->head_dim);
            rotary(k, p->head_dim, pos);
        }

        /** 
         * Multi-headed attention
         */
        attention(p, s, l, pos);

        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, all_heads_dim);
        matmul(s->xb2, &s->xq, w->wo + l, all_heads_dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++)
            x[i] += s->xb2[i];

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(&s->xq, s->xb, dim);
        matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

        // SwiGLU non-linearity
        swish(s, hidden_dim);

        // final matmul to get the output of the ffn
        quantize(&s->hq, s->hb, hidden_dim);
        matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++)
            x[i] += s->xb[i];
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    quantize(&s->xq, x, dim);
    matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char **vocab;
    float *merge_scores;
    int vocab_size;
    unsigned int max_token_length;
    unsigned int bos_token_id;
    unsigned int eos_token_id;
    char prompt_template[1024];
    char system_prompt_template[1024];
} Tokenizer;

void load_prompt_template(char *checkpoint_path, char *out_template, int with_system_prompt, int enable_thinking) {
    char prompt_path[1024];

    strcpy(prompt_path, checkpoint_path);
    if (with_system_prompt)
        strcat(prompt_path, enable_thinking ? ".template.with-system-and-thinking" : ".template.with-system");
    else
        strcat(prompt_path, enable_thinking ? ".template.with-thinking" : ".template");

    memset(out_template, 0, 1024);
    FILE *file = fopen(prompt_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't load prompt template %s\n", prompt_path); exit(EXIT_FAILURE); }
    fread(out_template, 1024, 1, file);
    fclose(file);
}

void build_tokenizer(Tokenizer *t, char *checkpoint_path, int vocab_size, int enable_thinking) {
    char tokenizer_path[1024];

    strcpy(tokenizer_path, checkpoint_path);
    strcat(tokenizer_path, ".tokenizer");

    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->merge_scores = (float *)malloc(vocab_size * sizeof(float));

    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't load tokenizer model %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    fread(&t->max_token_length, sizeof(int), 1, file);
    fread(&t->bos_token_id, sizeof(int), 1, file);
    fread(&t->eos_token_id, sizeof(int), 1, file);

    int len;

    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->merge_scores + i, sizeof(float), 1, file) != 1) {
            t->vocab[i] = (char *)malloc(1);
            t->vocab[i][0] = '\0'; // add the string terminating token
        } else {
            fread(&len, sizeof(int), 1, file);
            t->vocab[i] = (char *)malloc(len + 1);
            fread(t->vocab[i], 1, len, file);
            t->vocab[i][len] = '\0'; // add the string terminating token
        }
    }
    fclose(file);

    load_prompt_template(checkpoint_path, t->prompt_template, 0, enable_thinking);
    load_prompt_template(checkpoint_path, t->system_prompt_template, 1, enable_thinking);
}

void free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->merge_scores);
}

char *decode(Tokenizer *t, int token) {
    return t->vocab[token];
}

int str_lookup(char *str, char **vocab, int vocab_size) {
    // find a match for str in vocab, return its index or -1 if not found
    for (int i = 0; i < vocab_size; i++)
        if (!strcmp(str, vocab[i]))
            return i;

    return -1;
}

void encode(Tokenizer *t, char *text, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char *str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    char special_token[64 + 1];

    // start at 0 tokens
    *n_tokens = 0;

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {
        int id, found_special_token = 0;

        // set the buffer to the current byte
        str_buffer[0] = *c;
        str_buffer[1] = '\0';

        // special tokens begin with < and end with >. If we find a substring beginning with <
        // and ending with > and there's a token in the vocab for it, use that instead of parsing into
        // shorter tokens
        if (*c == '<') {
          int end_of_token_pos = -1;
          found_special_token = 0;
          for (int k = 0; *c != '\0' && k < 64; k++) {
              if (c[k] == '>') {
                  end_of_token_pos = k;
                  break;
              }
          }

          if (end_of_token_pos != -1) {
              strncpy(special_token, c, end_of_token_pos + 1);
              special_token[end_of_token_pos + 1] = 0;

              id = str_lookup(special_token, t->vocab, t->vocab_size);
              if (id != -1) {
                  c += end_of_token_pos;
                  found_special_token = 1;
              }
          }
        }

        // not a special token, just look up the single character
        if (!found_special_token)
            id = str_lookup(str_buffer, t->vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            printf("Warning: unknown character code point %d in input, skipping.\n", *str_buffer);
            (*n_tokens)++;
        }
    }

    // merge the best consecutive pair each iteration
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->vocab, t->vocab_size);

            if (id != -1 && t->merge_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->merge_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1)
            break; // we couldn't find any more pairs to merge, so we're done

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++)
            tokens[i] = tokens[i + 1];

        (*n_tokens)--; // token length decreased
    }

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex *probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf)
            return i;
    }
    return n - 1; // in case of rounding errors
}

int compare(const void *a, const void *b) {
    ProbIndex *a_ = (ProbIndex *) a;
    ProbIndex *b_ = (ProbIndex *) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf)
            return probindex[i].index;
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler *sampler, float *logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "Please provide a prompt using -i <string> on the command line.\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence

    while (pos < transformer->config.seq_len) {
        // forward the transformer to get logits for the next token
        float *logits = forward(transformer, token, pos);

        // advance the state state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // print the token as string, decode it with the Tokenizer object
        printf("%s", decode(tokenizer, token));
        fflush(stdout);
        token = next;

        // data-dependent terminating condition: the BOS token delimits sequences
        if (pos >= num_prompt_tokens && (next == tokenizer->bos_token_id || next == tokenizer->eos_token_id))
            break;
    }
    printf("\n");
    free(prompt_tokens);
}

void read_stdin(const char *guide, char *buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n')
            buffer[len - 1] = '\0'; // strip newline
    }
}

// ----------------------------------------------------------------------------
// chat loop

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *cli_user_prompt, char *system_prompt) {
    // buffers for reading the system prompt and user prompt from stdin
    char user_prompt[32768];
    char rendered_prompt[32768];
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc(32768 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int prev_token;
    int pos = 0;     // position in the sequence

    while (1) {
        // if context window is exceeded, clear it
        if (pos >= transformer->config.seq_len) {
            printf("\n(context window full, clearing)\n");
            user_turn = 1;
            pos = 0;
        }

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the user prompt
            if (cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                if (pos > 0)
                    break;
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("\n> ", user_prompt, sizeof(user_prompt));
                // terminate if user enters a blank prompt
                if (!user_prompt[0])
                    break;
            }

            // render user/system prompts into the Qwen3 prompt template schema
            if (pos == 0 && system_prompt)
                sprintf(rendered_prompt, tokenizer->system_prompt_template, system_prompt, user_prompt);
            else
                sprintf(rendered_prompt, tokenizer->prompt_template, user_prompt);

            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }

        // printf("|pos=%d token=%d '%s'|\n",pos,token,tokenizer->vocab[token]);

        // forward the transformer to get logits for the next token
        float *logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        // assistant is responding
        if (user_idx >= num_prompt_tokens) {
            if (token == tokenizer->bos_token_id || token == tokenizer->eos_token_id) {
                // EOS token ends the assistant turn
                printf("\n");
                user_turn = 1;
            } else if (next != tokenizer->bos_token_id && next != tokenizer->eos_token_id) {
                printf("%s", decode(tokenizer, next));
                fflush(stdout);
            }
        }
    }
    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// CLI

void error_usage() {
    fprintf(stderr, "Usage:   runq <checkpoint> [options]\n");
    fprintf(stderr, "Example: runq Qwen3-4B.bin -r 1\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1], default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -c <int>    context window size, 0 (default) = max_seq_len\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: chat\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -y <string> system prompt in chat mode, default is none\n");
    fprintf(stderr, "  -r <int>    reasoning mode, 0 (default) = no thinking, 1 = thinking\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    float temperature = 1.0f;   // 0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "chat";        // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode
    int enable_thinking = 0;    // 1 enables thinking
    int ctx_length = 0;         // context length

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'c') { ctx_length = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else if (argv[i][1] == 'r') { enable_thinking = atoi(argv[i + 1]); }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0) temperature = 0;
    if (topp < 0 || 1.0 < topp) topp = 0.9;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path, ctx_length);

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, checkpoint_path, transformer.config.vocab_size, enable_thinking);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    if (!prompt)
        printf("hidden_size=%d, intermediate_size=%d, num_hidden_layers=%d, num_attention_heads=%d, num_kv_heads=%d, head_dim=%d, ctx_length=%d, vocab_size=%d, shared_classifier=%d, quantization_block_size=%d\n", transformer.config.dim, transformer.config.hidden_dim, transformer.config.n_layers, transformer.config.n_heads, transformer.config.n_kv_heads, transformer.config.head_dim, transformer.config.seq_len, transformer.config.vocab_size, transformer.config.shared_classifier, transformer.config.group_size);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
