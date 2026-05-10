// Stage 1: same math as stage 0, but every intermediate buffer + every weight
// pointer is preallocated/precached at construction. forward() does ZERO heap
// allocation — verified by AllocGuard.
#include "engine.h"
#include <cmath>
#include <cstring>
#include <stdexcept>

// ---------- primitives (same naive math as stage 0) ----------

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

// ---------- offset planning ----------

static size_t aligned(size_t x, size_t a = 64) { return (x + a - 1) & ~(a - 1); }

static PoolOffsets plan_pool(const Stage1Config& cfg) {
    const int T = cfg.seq_len, D = cfg.d_model, H = cfg.n_heads, F = cfg.d_ff;
    const size_t fs = sizeof(float);
    PoolOffsets o{};
    size_t cur = 0;
    auto reserve = [&](size_t bytes) -> size_t { size_t off = cur; cur = aligned(cur + bytes); return off; };
    o.h      = reserve(T * D * fs);
    o.qkv    = reserve(T * 3 * D * fs);
    o.q      = reserve(T * D * fs);
    o.k      = reserve(T * D * fs);
    o.v      = reserve(T * D * fs);
    o.scores = reserve(H * T * T * fs);
    o.attn   = reserve(T * D * fs);
    o.tmp    = reserve(T * D * fs);
    o.ff     = reserve(T * F * fs);
    o.pooled = reserve(D * fs);
    o.total_bytes = cur;
    return o;
}

// ---------- engine ----------

Stage1Engine::Stage1Engine(const std::string& bin_path, Stage1Config cfg)
    : cfg_(cfg), weights_(bin_path),
      off_(plan_pool(cfg)),
      pool_(off_.total_bytes) {
    // Cache top-level pointers
    input_proj_w_ = weights_.get("input_proj.weight").fp32();
    input_proj_b_ = weights_.get("input_proj.bias").fp32();
    pos_          = weights_.get("pos").fp32();
    head_w_       = weights_.get("head.weight").fp32();
    head_b_       = weights_.get("head.bias").fp32();

    // Cache per-block pointers
    blocks_.resize(cfg_.n_layers);
    for (int L = 0; L < cfg_.n_layers; L++) {
        std::string p = "blocks." + std::to_string(L) + ".";
        BlockPtrs& b = blocks_[L];
        b.ln1_w      = weights_.get(p + "ln1.weight").fp32();
        b.ln1_b      = weights_.get(p + "ln1.bias").fp32();
        b.qkv_w      = weights_.get(p + "qkv.weight").fp32();
        b.qkv_b      = weights_.get(p + "qkv.bias").fp32();
        b.attn_out_w = weights_.get(p + "attn_out.weight").fp32();
        b.attn_out_b = weights_.get(p + "attn_out.bias").fp32();
        b.ln2_w      = weights_.get(p + "ln2.weight").fp32();
        b.ln2_b      = weights_.get(p + "ln2.bias").fp32();
        b.ff1_w      = weights_.get(p + "ff1.weight").fp32();
        b.ff1_b      = weights_.get(p + "ff1.bias").fp32();
        b.ff2_w      = weights_.get(p + "ff2.weight").fp32();
        b.ff2_b      = weights_.get(p + "ff2.bias").fp32();
    }
}

void Stage1Engine::forward(const float* x, float* out) {
    const int T = cfg_.seq_len, D = cfg_.d_model, H = cfg_.n_heads, Dh = D / H, F = cfg_.d_ff;

    float* h      = pool_.view(off_.h);
    float* qkv    = pool_.view(off_.qkv);
    float* q      = pool_.view(off_.q);
    float* k      = pool_.view(off_.k);
    float* v      = pool_.view(off_.v);
    float* scores = pool_.view(off_.scores);
    float* attn   = pool_.view(off_.attn);
    float* tmp    = pool_.view(off_.tmp);
    float* ff     = pool_.view(off_.ff);
    float* pooled = pool_.view(off_.pooled);

    // Input projection + positional encoding
    linear(x, input_proj_w_, input_proj_b_, h, T, cfg_.n_vars, D);
    for (int i = 0; i < T*D; i++) h[i] += pos_[i];

    for (int L = 0; L < cfg_.n_layers; L++) {
        const BlockPtrs& bp = blocks_[L];

        // Self-attention
        layer_norm(h, bp.ln1_w, bp.ln1_b, tmp, T, D);
        linear(tmp, bp.qkv_w, bp.qkv_b, qkv, T, D, 3*D);
        for (int t = 0; t < T; t++) {
            std::memcpy(q + t*D, qkv + t*3*D + 0*D, D*sizeof(float));
            std::memcpy(k + t*D, qkv + t*3*D + 1*D, D*sizeof(float));
            std::memcpy(v + t*D, qkv + t*3*D + 2*D, D*sizeof(float));
        }

        std::memset(attn, 0, T*D*sizeof(float));
        const float scale = 1.f / std::sqrt((float)Dh);
        for (int hi = 0; hi < H; hi++) {
            float* sc = scores + hi*T*T;
            for (int i = 0; i < T; i++)
                for (int j = 0; j < T; j++) {
                    float s = 0.f;
                    for (int d = 0; d < Dh; d++) s += q[i*D + hi*Dh + d] * k[j*D + hi*Dh + d];
                    sc[i*T + j] = s * scale;
                }
            softmax_rows(sc, T, T);
            for (int i = 0; i < T; i++)
                for (int d = 0; d < Dh; d++) {
                    float s = 0.f;
                    for (int j = 0; j < T; j++) s += sc[i*T + j] * v[j*D + hi*Dh + d];
                    attn[i*D + hi*Dh + d] = s;
                }
        }

        linear(attn, bp.attn_out_w, bp.attn_out_b, tmp, T, D, D);
        for (int i = 0; i < T*D; i++) h[i] += tmp[i];

        // FFN
        layer_norm(h, bp.ln2_w, bp.ln2_b, tmp, T, D);
        linear(tmp, bp.ff1_w, bp.ff1_b, ff, T, D, F);
        for (int i = 0; i < T*F; i++) ff[i] = gelu_tanh(ff[i]);
        linear(ff, bp.ff2_w, bp.ff2_b, tmp, T, F, D);
        for (int i = 0; i < T*D; i++) h[i] += tmp[i];
    }

    // Mean pool over time
    std::memset(pooled, 0, D*sizeof(float));
    for (int t = 0; t < T; t++) for (int d = 0; d < D; d++) pooled[d] += h[t*D + d];
    for (int d = 0; d < D; d++) pooled[d] /= T;

    // Head
    linear(pooled, head_w_, head_b_, out, 1, D, cfg_.horizon * cfg_.n_vars);
}
