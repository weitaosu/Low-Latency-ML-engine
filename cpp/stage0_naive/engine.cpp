#include "engine.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <stdexcept>

// ---------- naive primitives ----------

static void matmul(const float* A, const float* B, float* C,
                   int M, int K, int N) {
    // C[M,N] = A[M,K] @ B[K,N]    (PyTorch Linear stores W as [out, in] = [N, K])
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float s = 0.f;
            for (int k = 0; k < K; k++) s += A[m*K + k] * B[k*N + n];
            C[m*N + n] = s;
        }
    }
}

// y = x @ W^T + b    where W is [out, in] (PyTorch convention)
static void linear(const float* x, const float* W, const float* b,
                   float* y, int M, int in_dim, int out_dim) {
    for (int m = 0; m < M; m++) {
        for (int o = 0; o < out_dim; o++) {
            float s = b ? b[o] : 0.f;
            for (int k = 0; k < in_dim; k++) s += x[m*in_dim + k] * W[o*in_dim + k];
            y[m*out_dim + o] = s;
        }
    }
}

static void layer_norm(const float* x, const float* g, const float* b,
                       float* y, int M, int D, float eps = 1e-5f) {
    for (int m = 0; m < M; m++) {
        float mean = 0.f;
        for (int d = 0; d < D; d++) mean += x[m*D + d];
        mean /= D;
        float var = 0.f;
        for (int d = 0; d < D; d++) { float v = x[m*D + d] - mean; var += v*v; }
        var /= D;
        float inv = 1.f / std::sqrt(var + eps);
        for (int d = 0; d < D; d++)
            y[m*D + d] = (x[m*D + d] - mean) * inv * g[d] + b[d];
    }
}

static inline float gelu_tanh(float x) {
    // matches torch.nn.GELU(approximate='tanh')
    constexpr float c0 = 0.044715f;
    constexpr float c1 = 0.7978845608f;  // sqrt(2/pi)
    float t = c1 * (x + c0 * x * x * x);
    return 0.5f * x * (1.f + std::tanh(t));
}

static void softmax_rows(float* x, int M, int N) {
    for (int m = 0; m < M; m++) {
        float* row = x + m*N;
        float mx = row[0];
        for (int n = 1; n < N; n++) if (row[n] > mx) mx = row[n];
        float s = 0.f;
        for (int n = 0; n < N; n++) { row[n] = std::exp(row[n] - mx); s += row[n]; }
        float inv = 1.f / s;
        for (int n = 0; n < N; n++) row[n] *= inv;
    }
}

// ---------- Stage 0 engine ----------

Stage0Engine::Stage0Engine(const std::string& bin_path, Stage0Config cfg)
    : cfg_(cfg), weights_(bin_path) {}

void Stage0Engine::forward(const float* x, float* out) {
    const int T = cfg_.seq_len;
    const int D = cfg_.d_model;
    const int H = cfg_.n_heads;
    const int Dh = D / H;
    const int F = cfg_.d_ff;

    // Buffers (naive: malloc per call; preallocation comes in stage 1)
    std::vector<float> h(T * D), pos(T * D);
    std::vector<float> tmp(T * D), tmp2(T * D);
    std::vector<float> qkv(T * 3 * D);
    std::vector<float> q(T * D), k(T * D), v(T * D);
    std::vector<float> scores(H * T * T);
    std::vector<float> attn(T * D);
    std::vector<float> ff(T * F);

    // Input projection: x [T, n_vars] @ W^T + b → h [T, D]
    const float* W_in = weights_.get("input_proj.weight").fp32();
    const float* b_in = weights_.get("input_proj.bias").fp32();
    linear(x, W_in, b_in, h.data(), T, cfg_.n_vars, D);

    // Add positional encoding
    const float* PE = weights_.get("pos").fp32();
    for (int i = 0; i < T*D; i++) h[i] += PE[i];

    // Encoder blocks
    for (int L = 0; L < cfg_.n_layers; L++) {
        std::string p = "blocks." + std::to_string(L) + ".";
        // ---- self-attention sub-block ----
        layer_norm(h.data(), weights_.get(p+"ln1.weight").fp32(),
                   weights_.get(p+"ln1.bias").fp32(), tmp.data(), T, D);
        linear(tmp.data(),
               weights_.get(p+"qkv.weight").fp32(),
               weights_.get(p+"qkv.bias").fp32(),
               qkv.data(), T, D, 3*D);
        // Split q,k,v
        for (int t = 0; t < T; t++) {
            std::memcpy(q.data() + t*D, qkv.data() + t*3*D + 0*D, D*sizeof(float));
            std::memcpy(k.data() + t*D, qkv.data() + t*3*D + 1*D, D*sizeof(float));
            std::memcpy(v.data() + t*D, qkv.data() + t*3*D + 2*D, D*sizeof(float));
        }

        // For each head, compute scores = (Q @ K^T) / sqrt(Dh); softmax; @ V
        std::memset(attn.data(), 0, T*D*sizeof(float));
        const float scale = 1.f / std::sqrt((float)Dh);
        for (int h_idx = 0; h_idx < H; h_idx++) {
            float* sc = scores.data() + h_idx*T*T;
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < T; j++) {
                    float s = 0.f;
                    for (int d = 0; d < Dh; d++)
                        s += q[i*D + h_idx*Dh + d] * k[j*D + h_idx*Dh + d];
                    sc[i*T + j] = s * scale;
                }
            }
            softmax_rows(sc, T, T);
            for (int i = 0; i < T; i++) {
                for (int d = 0; d < Dh; d++) {
                    float s = 0.f;
                    for (int j = 0; j < T; j++) s += sc[i*T + j] * v[j*D + h_idx*Dh + d];
                    attn[i*D + h_idx*Dh + d] = s;
                }
            }
        }

        // attn_out projection + residual
        linear(attn.data(),
               weights_.get(p+"attn_out.weight").fp32(),
               weights_.get(p+"attn_out.bias").fp32(),
               tmp.data(), T, D, D);
        for (int i = 0; i < T*D; i++) h[i] += tmp[i];

        // ---- FFN sub-block ----
        layer_norm(h.data(), weights_.get(p+"ln2.weight").fp32(),
                   weights_.get(p+"ln2.bias").fp32(), tmp.data(), T, D);
        linear(tmp.data(),
               weights_.get(p+"ff1.weight").fp32(),
               weights_.get(p+"ff1.bias").fp32(),
               ff.data(), T, D, F);
        for (int i = 0; i < T*F; i++) ff[i] = gelu_tanh(ff[i]);
        linear(ff.data(),
               weights_.get(p+"ff2.weight").fp32(),
               weights_.get(p+"ff2.bias").fp32(),
               tmp.data(), T, F, D);
        for (int i = 0; i < T*D; i++) h[i] += tmp[i];
    }

    // Mean-pool over time → [D]
    std::vector<float> pooled(D, 0.f);
    for (int t = 0; t < T; t++) for (int d = 0; d < D; d++) pooled[d] += h[t*D + d];
    for (int d = 0; d < D; d++) pooled[d] /= T;

    // Head: pooled [D] @ head.weight^T + head.bias → [horizon * n_vars]
    linear(pooled.data(),
           weights_.get("head.weight").fp32(),
           weights_.get("head.bias").fp32(),
           out, 1, D, cfg_.horizon * cfg_.n_vars);
}
