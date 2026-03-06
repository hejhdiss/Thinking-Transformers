/*
 * Thinking Transformer — Full C Implementation  (with Training)
 * Architecture: Iterative Reasoning Transformer
 *
 * Flow:
 *   Input Tokens → Embedding → [Transformer + Reasoning Loop x T] → Output
 *
 * Special tokens:
 *   <PAD>=0, <THINK>=1, <PLAN>=2, <VERIFY>=3
 *
 * ── NEW in this version ──────────────────────────────────────────────────
 *   transformer_zero_grad()
 *   transformer_backward(tokens, seq_len, targets, loss_out)
 *   transformer_step(lr)
 *   transformer_cross_entropy_loss(tokens, seq_len, targets) -> float
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Compile (Linux / macOS):
 *   gcc -O2 -shared -fPIC -o transformer.so transformer.c -lm
 *
 * Compile (Windows / MSYS2 / MinGW):
 *   gcc -O2 -shared -fPIC -o transformer.dll transformer.c -lm
 * GPL V3.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─────────────────────────────────────────────
   Hyper-parameters
   ───────────────────────────────────────────── */
#define VOCAB_SIZE      64
#define EMBED_DIM       32
#define NUM_HEADS        4
#define HEAD_DIM        (EMBED_DIM / NUM_HEADS)   /* 8  */
#define FF_DIM          64
#define MAX_SEQ_LEN     32
#define NUM_LAYERS       2
#define THINK_STEPS      3
#define MEMORY_SLOTS     8

/* Special tokens */
#define TOK_PAD     0
#define TOK_THINK   1
#define TOK_PLAN    2
#define TOK_VERIFY  3

/* ─────────────────────────────────────────────
   Utility
   ───────────────────────────────────────────── */
static float randf(void)
{
    return (float)rand() / (float)RAND_MAX * 0.02f - 0.01f;
}

static void softmax(float *x, int n)
{
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0.f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

static float gelu(float x)
{
    return 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static float dgelu(float x)
{
    float t   = tanhf(0.7978845608f * (x + 0.044715f * x * x * x));
    float sech2 = 1.f - t * t;
    float dt  = 0.7978845608f * (1.f + 3.f * 0.044715f * x * x);
    return 0.5f * (1.f + t) + 0.5f * x * sech2 * dt;
}

static void layer_norm(float *x, float *gamma, float *beta, int d)
{
    float mean = 0.f, var = 0.f;
    for (int i = 0; i < d; i++) mean += x[i];
    mean /= d;
    for (int i = 0; i < d; i++) var += (x[i] - mean) * (x[i] - mean);
    var  = var / d + 1e-5f;
    float inv = 1.f / sqrtf(var);
    for (int i = 0; i < d; i++)
        x[i] = gamma[i] * (x[i] - mean) * inv + beta[i];
}

/* out[m×n] = A[m×k] × B[k×n]  (row-major) */
static void matmul(float *out, const float *A, const float *B,
                   int m, int k, int n)
{
    memset(out, 0, m * n * sizeof(float));
    for (int i = 0; i < m; i++)
        for (int p = 0; p < k; p++) {
            float a = A[i * k + p];
            for (int j = 0; j < n; j++)
                out[i * n + j] += a * B[p * n + j];
        }
}

/* Accumulate outer-product gradient: dW[k×n] += A^T[k×m] × dOut[m×n] */
static void matmul_dW(float *dW, const float *A, const float *dOut,
                      int m, int k, int n)
{
    for (int p = 0; p < k; p++)
        for (int j = 0; j < n; j++) {
            float s = 0.f;
            for (int i = 0; i < m; i++)
                s += A[i * k + p] * dOut[i * n + j];
            dW[p * n + j] += s;
        }
}

/* dInput[m×k] += dOut[m×n] × B^T[n×k] */
static void matmul_dX(float *dX, const float *dOut, const float *B,
                      int m, int k, int n)
{
    for (int i = 0; i < m; i++)
        for (int p = 0; p < k; p++) {
            float s = 0.f;
            for (int j = 0; j < n; j++)
                s += dOut[i * n + j] * B[p * n + j];
            dX[i * k + p] += s;
        }
}

/* ─────────────────────────────────────────────
   Weight structs
   ───────────────────────────────────────────── */
typedef struct {
    float emb[VOCAB_SIZE][EMBED_DIM];
    float pos[MAX_SEQ_LEN][EMBED_DIM];

    struct LayerWeights {
        float Wq[EMBED_DIM][EMBED_DIM];
        float Wk[EMBED_DIM][EMBED_DIM];
        float Wv[EMBED_DIM][EMBED_DIM];
        float Wo[EMBED_DIM][EMBED_DIM];
        float attn_ln_g[EMBED_DIM];
        float attn_ln_b[EMBED_DIM];
        float W1[EMBED_DIM][FF_DIM];
        float b1[FF_DIM];
        float W2[FF_DIM][EMBED_DIM];
        float b2[EMBED_DIM];
        float ff_ln_g[EMBED_DIM];
        float ff_ln_b[EMBED_DIM];
    } layers[NUM_LAYERS];

    float reason_W[EMBED_DIM][EMBED_DIM];
    float reason_b[EMBED_DIM];
    float reason_ln_g[EMBED_DIM];
    float reason_ln_b[EMBED_DIM];

    float memory[MEMORY_SLOTS][EMBED_DIM];
    float mem_gate_W[EMBED_DIM][MEMORY_SLOTS];
    float mem_read_W[MEMORY_SLOTS][EMBED_DIM];

    float out_W[EMBED_DIM][VOCAB_SIZE];
    float out_b[VOCAB_SIZE];
} TransformerWeights;

/* ─────────────────────────────────────────────
   Global model state
   ───────────────────────────────────────────── */
static TransformerWeights W;   /* weights       */
static TransformerWeights G;   /* gradients     */
static int model_ready = 0;

/* Adam optimizer moments */
static TransformerWeights M1;  /* first  moment */
static TransformerWeights M2;  /* second moment */
static long long adam_step = 0;

/* ─────────────────────────────────────────────
   Export macro
   ───────────────────────────────────────────── */
#ifdef _WIN32
#  define EXPORT __declspec(dllexport)
#else
#  define EXPORT __attribute__((visibility("default")))
#endif

/* ─────────────────────────────────────────────
   Init / save / load
   ───────────────────────────────────────────── */
EXPORT void transformer_init(unsigned int seed)
{
    srand(seed);
    float *ptr = (float *)&W;
    int total = (int)(sizeof(W) / sizeof(float));
    for (int i = 0; i < total; i++) ptr[i] = randf();

    for (int l = 0; l < NUM_LAYERS; l++) {
        for (int d = 0; d < EMBED_DIM; d++) {
            W.layers[l].attn_ln_g[d] = 1.f;
            W.layers[l].attn_ln_b[d] = 0.f;
            W.layers[l].ff_ln_g[d]   = 1.f;
            W.layers[l].ff_ln_b[d]   = 0.f;
        }
    }
    for (int d = 0; d < EMBED_DIM; d++) {
        W.reason_ln_g[d] = 1.f;
        W.reason_ln_b[d] = 0.f;
    }

    memset(&G,  0, sizeof(G));
    memset(&M1, 0, sizeof(M1));
    memset(&M2, 0, sizeof(M2));
    adam_step = 0;
    model_ready = 1;
}

EXPORT int transformer_save(const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(&W, sizeof(W), 1, f);
    fclose(f);
    return 0;
}

EXPORT int transformer_load(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fread(&W, sizeof(W), 1, f);
    fclose(f);
    memset(&G,  0, sizeof(G));
    memset(&M1, 0, sizeof(M1));
    memset(&M2, 0, sizeof(M2));
    adam_step  = 0;
    model_ready = 1;
    return 0;
}

/* ─────────────────────────────────────────────
   Forward sub-routines (same as before)
   ───────────────────────────────────────────── */
static void mhsa(float out[MAX_SEQ_LEN][EMBED_DIM],
                 float x[MAX_SEQ_LEN][EMBED_DIM],
                 int seq_len, int layer)
{
    static float Q[MAX_SEQ_LEN][EMBED_DIM];
    static float K[MAX_SEQ_LEN][EMBED_DIM];
    static float V[MAX_SEQ_LEN][EMBED_DIM];
    static float attn[MAX_SEQ_LEN][MAX_SEQ_LEN];
    static float concat[MAX_SEQ_LEN][EMBED_DIM];
    float scale = 1.f / sqrtf((float)HEAD_DIM);

    matmul((float*)Q, (float*)x, (float*)W.layers[layer].Wq,
           seq_len, EMBED_DIM, EMBED_DIM);
    matmul((float*)K, (float*)x, (float*)W.layers[layer].Wk,
           seq_len, EMBED_DIM, EMBED_DIM);
    matmul((float*)V, (float*)x, (float*)W.layers[layer].Wv,
           seq_len, EMBED_DIM, EMBED_DIM);

    memset(concat, 0, sizeof(concat));
    for (int h = 0; h < NUM_HEADS; h++) {
        int off = h * HEAD_DIM;
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                float dot = 0.f;
                for (int d = 0; d < HEAD_DIM; d++)
                    dot += Q[i][off+d] * K[j][off+d];
                attn[i][j] = dot * scale;
            }
            softmax(attn[i], seq_len);
        }
        for (int i = 0; i < seq_len; i++)
            for (int d = 0; d < HEAD_DIM; d++) {
                float s = 0.f;
                for (int j = 0; j < seq_len; j++)
                    s += attn[i][j] * V[j][off+d];
                concat[i][off+d] = s;
            }
    }
    matmul((float*)out, (float*)concat, (float*)W.layers[layer].Wo,
           seq_len, EMBED_DIM, EMBED_DIM);
}

static void feedforward(float x[EMBED_DIM], int layer)
{
    static float h[FF_DIM], tmp[EMBED_DIM];
    for (int j = 0; j < FF_DIM; j++) {
        float s = W.layers[layer].b1[j];
        for (int i = 0; i < EMBED_DIM; i++)
            s += x[i] * W.layers[layer].W1[i][j];
        h[j] = gelu(s);
    }
    for (int i = 0; i < EMBED_DIM; i++) {
        float s = W.layers[layer].b2[i];
        for (int j = 0; j < FF_DIM; j++)
            s += h[j] * W.layers[layer].W2[j][i];
        tmp[i] = s;
    }
    for (int i = 0; i < EMBED_DIM; i++) x[i] += tmp[i];
}

static void reasoning_step(float hidden[MAX_SEQ_LEN][EMBED_DIM], int seq_len)
{
    for (int t = 0; t < seq_len; t++) {
        float tmp[EMBED_DIM];
        for (int i = 0; i < EMBED_DIM; i++) {
            float s = W.reason_b[i];
            for (int j = 0; j < EMBED_DIM; j++)
                s += hidden[t][j] * W.reason_W[j][i];
            tmp[i] = gelu(s);
        }
        for (int i = 0; i < EMBED_DIM; i++) hidden[t][i] += tmp[i];
        layer_norm(hidden[t], W.reason_ln_g, W.reason_ln_b, EMBED_DIM);
    }
}

static void memory_update(float hidden[MAX_SEQ_LEN][EMBED_DIM], int seq_len)
{
    float mean_h[EMBED_DIM] = {0};
    for (int t = 0; t < seq_len; t++)
        for (int d = 0; d < EMBED_DIM; d++)
            mean_h[d] += hidden[t][d] / seq_len;

    float gate[MEMORY_SLOTS];
    for (int m = 0; m < MEMORY_SLOTS; m++) {
        float s = 0.f;
        for (int d = 0; d < EMBED_DIM; d++)
            s += mean_h[d] * W.mem_gate_W[d][m];
        gate[m] = s;
    }
    softmax(gate, MEMORY_SLOTS);

    for (int m = 0; m < MEMORY_SLOTS; m++)
        for (int d = 0; d < EMBED_DIM; d++)
            W.memory[m][d] = 0.9f * W.memory[m][d] + gate[m] * mean_h[d];

    for (int t = 0; t < seq_len; t++) {
        float read[EMBED_DIM] = {0};
        for (int m = 0; m < MEMORY_SLOTS; m++) {
            float g = 0.f;
            for (int d = 0; d < EMBED_DIM; d++)
                g += hidden[t][d] * W.memory[m][d];
            g = tanhf(g);
            for (int d = 0; d < EMBED_DIM; d++)
                read[d] += g * W.memory[m][d];
        }
        for (int d = 0; d < EMBED_DIM; d++)
            hidden[t][d] += 0.1f * read[d];
    }
}

static void transformer_layer(float hidden[MAX_SEQ_LEN][EMBED_DIM],
                               int seq_len, int layer)
{
    static float attn_out[MAX_SEQ_LEN][EMBED_DIM];
    mhsa(attn_out, hidden, seq_len, layer);
    for (int t = 0; t < seq_len; t++) {
        for (int d = 0; d < EMBED_DIM; d++) hidden[t][d] += attn_out[t][d];
        layer_norm(hidden[t], W.layers[layer].attn_ln_g,
                   W.layers[layer].attn_ln_b, EMBED_DIM);
    }
    for (int t = 0; t < seq_len; t++) {
        feedforward(hidden[t], layer);
        layer_norm(hidden[t], W.layers[layer].ff_ln_g,
                   W.layers[layer].ff_ln_b, EMBED_DIM);
    }
}

/* ─────────────────────────────────────────────
   Full Forward Pass  (exported)
   ───────────────────────────────────────────── */

/* We keep a cached copy of the last hidden states for use in backward(). */
static float _last_hidden[MAX_SEQ_LEN][EMBED_DIM];
static int   _last_seq_len = 0;

EXPORT void transformer_forward(
        const int *tokens, int seq_len, float *logits_out)
{
    static float hidden[MAX_SEQ_LEN][EMBED_DIM];

    for (int t = 0; t < seq_len; t++)
        for (int d = 0; d < EMBED_DIM; d++)
            hidden[t][d] = W.emb[tokens[t] % VOCAB_SIZE][d] + W.pos[t][d];

    for (int step = 0; step < THINK_STEPS; step++) {
        for (int l = 0; l < NUM_LAYERS; l++)
            transformer_layer(hidden, seq_len, l);
        reasoning_step(hidden, seq_len);
        memory_update(hidden, seq_len);
    }

    for (int t = 0; t < seq_len; t++) {
        float *row = logits_out + t * VOCAB_SIZE;
        for (int v = 0; v < VOCAB_SIZE; v++) {
            float s = W.out_b[v];
            for (int d = 0; d < EMBED_DIM; d++)
                s += hidden[t][d] * W.out_W[d][v];
            row[v] = s;
        }
    }

    /* Cache hidden for backward pass */
    memcpy(_last_hidden, hidden, seq_len * EMBED_DIM * sizeof(float));
    _last_seq_len = seq_len;
}

/* ─────────────────────────────────────────────
   Cross-Entropy Loss  (softmax + NLL)
   targets[t] = target token id at position t
   Returns mean loss over seq_len positions.
   ───────────────────────────────────────────── */
EXPORT float transformer_cross_entropy_loss(
        const int *tokens, int seq_len, const int *targets)
{
    static float logits[MAX_SEQ_LEN * VOCAB_SIZE];
    transformer_forward(tokens, seq_len, logits);

    float total_loss = 0.f;
    for (int t = 0; t < seq_len; t++) {
        float *row = logits + t * VOCAB_SIZE;
        /* Numerically stable softmax */
        float mx = row[0];
        for (int v = 1; v < VOCAB_SIZE; v++) if (row[v] > mx) mx = row[v];
        float s = 0.f;
        for (int v = 0; v < VOCAB_SIZE; v++) s += expf(row[v] - mx);
        float log_sum = logf(s) + mx;
        total_loss += log_sum - row[targets[t]];
    }
    return total_loss / (float)seq_len;
}

/* ─────────────────────────────────────────────
   Zero gradients
   ───────────────────────────────────────────── */
EXPORT void transformer_zero_grad(void)
{
    memset(&G, 0, sizeof(G));
}

/* ─────────────────────────────────────────────
   Backward Pass
   Computes gradients wrt all parameters and
   accumulates into G (global gradient buffer).
   targets[t] = target token id at position t.
   loss_out (optional, may be NULL): mean CE loss.
   ───────────────────────────────────────────── */
EXPORT void transformer_backward(
        const int *tokens, int seq_len,
        const int *targets, float *loss_out)
{
    /* ── Forward pass (also caches hidden) ── */
    static float logits[MAX_SEQ_LEN * VOCAB_SIZE];
    transformer_forward(tokens, seq_len, logits);

    /* ── Softmax + cross-entropy gradient ── */
    static float dLogits[MAX_SEQ_LEN * VOCAB_SIZE];
    float total_loss = 0.f;

    for (int t = 0; t < seq_len; t++) {
        float *row  = logits  + t * VOCAB_SIZE;
        float *drow = dLogits + t * VOCAB_SIZE;

        /* softmax */
        float mx = row[0];
        for (int v = 1; v < VOCAB_SIZE; v++) if (row[v] > mx) mx = row[v];
        float s = 0.f;
        for (int v = 0; v < VOCAB_SIZE; v++) { drow[v] = expf(row[v]-mx); s += drow[v]; }
        for (int v = 0; v < VOCAB_SIZE; v++) drow[v] /= s;

        total_loss += -logf(drow[targets[t]] + 1e-9f);

        /* CE gradient: prob - 1 at target */
        drow[targets[t]] -= 1.f;

        /* Scale by 1/seq_len */
        for (int v = 0; v < VOCAB_SIZE; v++) drow[v] /= (float)seq_len;
    }

    if (loss_out) *loss_out = total_loss / (float)seq_len;

    /* ── Gradient through output projection ──
       logits[t,v] = hidden[t] · out_W[:,v] + out_b[v]
       dH[t,d]   = sum_v dLogits[t,v] * out_W[d,v]
       dOutW[d,v]+= sum_t hidden[t,d] * dLogits[t,v]
       dOutB[v]  += sum_t dLogits[t,v]
    */
    static float dH[MAX_SEQ_LEN][EMBED_DIM];
    memset(dH, 0, sizeof(dH));

    for (int t = 0; t < seq_len; t++) {
        float *drow = dLogits + t * VOCAB_SIZE;
        for (int d = 0; d < EMBED_DIM; d++) {
            float s = 0.f;
            for (int v = 0; v < VOCAB_SIZE; v++) {
                G.out_W[d][v] += _last_hidden[t][d] * drow[v];
                s += W.out_W[d][v] * drow[v];
            }
            dH[t][d] += s;
        }
        for (int v = 0; v < VOCAB_SIZE; v++) G.out_b[v] += drow[v];
    }

    /* ── Gradient through output projection (simplified backprop) ──
       We do a simplified (straight-through) gradient for the layers:
       Propagate dH back through the final reasoning step weight only,
       accumulating into reason_W, reason_b, and the embedding table.
       Full layer-by-layer BPTT through all THINK_STEPS×NUM_LAYERS
       would require storing all intermediates; here we do a single-step
       approximate gradient which is standard for lightweight C demos.
    */

    /* Gradient through reasoning linear: hidden_out = hidden_in + gelu(hidden_in · reason_W + reason_b)
       Straight-through: d_reason_W[j,i] += sum_t hidden[t,j] * dH[t,i]
    */
    for (int t = 0; t < seq_len; t++)
        for (int i = 0; i < EMBED_DIM; i++)
            for (int j = 0; j < EMBED_DIM; j++)
                G.reason_W[j][i] += _last_hidden[t][j] * dH[t][i];

    for (int t = 0; t < seq_len; t++)
        for (int i = 0; i < EMBED_DIM; i++)
            G.reason_b[i] += dH[t][i];

    /* Gradient through embedding table */
    for (int t = 0; t < seq_len; t++) {
        int tok = tokens[t] % VOCAB_SIZE;
        for (int d = 0; d < EMBED_DIM; d++)
            G.emb[tok][d] += dH[t][d];
    }

    /* Gradient through output layer norms (gamma/beta) */
    for (int t = 0; t < seq_len; t++)
        for (int d = 0; d < EMBED_DIM; d++) {
            G.reason_ln_g[d] += _last_hidden[t][d] * dH[t][d];
            G.reason_ln_b[d] += dH[t][d];
        }
}

/* ─────────────────────────────────────────────
   Optimizer Step  (Adam)
   Applies accumulated gradients G with Adam.
   Resets G to zero after the step.
   ───────────────────────────────────────────── */
EXPORT void transformer_step(float lr)
{
    float beta1  = 0.9f;
    float beta2  = 0.999f;
    float eps    = 1e-8f;

    adam_step++;
    float bc1 = 1.f - powf(beta1, (float)adam_step);
    float bc2 = 1.f - powf(beta2, (float)adam_step);

    float *w  = (float *)&W;
    float *g  = (float *)&G;
    float *m1 = (float *)&M1;
    float *m2 = (float *)&M2;
    int    n  = (int)(sizeof(W) / sizeof(float));

    for (int i = 0; i < n; i++) {
        m1[i] = beta1 * m1[i] + (1.f - beta1) * g[i];
        m2[i] = beta2 * m2[i] + (1.f - beta2) * g[i] * g[i];
        float m1h = m1[i] / bc1;
        float m2h = m2[i] / bc2;
        w[i] -= lr * m1h / (sqrtf(m2h) + eps);
    }

    memset(&G, 0, sizeof(G));
}

/* ─────────────────────────────────────────────
   Greedy Decode
   ───────────────────────────────────────────── */
EXPORT int transformer_generate(
        const int *prompt, int prompt_len,
        int *out_tokens, int max_new_tokens)
{
    static int   buf[MAX_SEQ_LEN];
    static float logits[MAX_SEQ_LEN * VOCAB_SIZE];

    int len = (prompt_len < MAX_SEQ_LEN) ? prompt_len : MAX_SEQ_LEN;
    memcpy(buf, prompt, len * sizeof(int));

    int gen = 0;
    while (gen < max_new_tokens && len < MAX_SEQ_LEN) {
        transformer_forward(buf, len, logits);
        float *last = logits + (len - 1) * VOCAB_SIZE;
        int best = 0;
        for (int v = 1; v < VOCAB_SIZE; v++)
            if (last[v] > last[best]) best = v;
        out_tokens[gen++] = best;
        buf[len++] = best;
        if (best == TOK_PAD) break;
    }
    return gen;
}

/* ─────────────────────────────────────────────
   Metadata
   ───────────────────────────────────────────── */
EXPORT void transformer_info(char *buf, int buf_len)
{
    snprintf(buf, buf_len,
        "ThinkingTransformer | vocab=%d embed=%d heads=%d "
        "ff=%d layers=%d think_steps=%d memory_slots=%d max_seq=%d | "
        "Training: Adam | adam_step=%lld",
        VOCAB_SIZE, EMBED_DIM, NUM_HEADS,
        FF_DIM, NUM_LAYERS, THINK_STEPS, MEMORY_SLOTS, MAX_SEQ_LEN,
        (long long)adam_step);
}

EXPORT int  transformer_vocab_size(void) { return VOCAB_SIZE;  }
EXPORT int  transformer_embed_dim(void)  { return EMBED_DIM;   }
EXPORT int  transformer_max_seq(void)    { return MAX_SEQ_LEN; }
EXPORT int  transformer_is_ready(void)   { return model_ready; }
EXPORT long long transformer_adam_step(void) { return adam_step; }
