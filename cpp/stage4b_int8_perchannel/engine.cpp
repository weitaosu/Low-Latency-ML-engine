// Stage 4b: per-channel weight quantization. Same math as 4a but w_scale
// is indexed by output channel (allowing tighter range per output column).
//
// y[o] = (acc - a_zp * sum_W[o]) * (a_scale * w_scale[o]) + bias[o]
//
// Targets per guide §8: 4b MSE within 1-5% of stage 3 FP32. 4a expected to be
// 1-2% worse than 4b due to coarser per-tensor weight scale.
#include "engine.h"
#include <cmath>
#include <cstring>
#include <stdexcept>

static void linear_fp32(const float* x, const float* W, const float* b,
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
    constexpr float c1 = 0.7978845608f;
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

static inline uint8_t quantize_one(float x, float a_scale, int a_zp) {
    int q = (int)std::lround(x / a_scale) + a_zp;
    if (q < 0) q = 0;
    else if (q > 255) q = 255;
    return (uint8_t)q;
}

static void quantized_linear_4b(const float* x, const QLinearParams& p,
                                float* y, int M, uint8_t* xq_scratch) {
    const int K = p.in_dim, N = p.out_dim;
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) xq_scratch[k] = quantize_one(x[m*K + k], p.a_scale, p.a_zp);
        for (int o = 0; o < N; o++) {
            int32_t acc = 0;
            const int8_t* Wo = p.W_q + o*K;
            for (int k = 0; k < K; k++) acc += (int32_t)xq_scratch[k] * (int32_t)Wo[k];
            float yv = (float)(acc - p.a_zp * p.sum_W[o]) * (p.a_scale * p.w_scale[o]);
            if (p.bias) yv += p.bias[o];
            y[m*N + o] = yv;
        }
    }
}

static size_t aligned(size_t x, size_t a = 64) { return (x + a - 1) & ~(a - 1); }

static PoolOffsets plan_pool(const Stage4bConfig& cfg) {
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
    o.xq     = reserve(F * sizeof(uint8_t));
    o.total_bytes = cur;
    return o;
}

static QLinearParams make_q(const WeightLoader& W, const std::string& base, int out_dim, int in_dim) {
    QLinearParams p;
    p.W_q     = (const int8_t*)  W.get(base + ".weight_q").data;
    p.sum_W   = (const int32_t*) W.get(base + ".sum_W").data;
    p.bias    = W.has(base + ".bias") ? (const float*) W.get(base + ".bias").data : nullptr;
    p.a_scale = ((const float*) W.get(base + ".a_scale").data)[0];
    p.a_zp    = (int) ((const float*) W.get(base + ".a_zp").data)[0];
    p.w_scale = (const float*) W.get(base + ".w_scale").data;     // per-channel vector
    p.out_dim = out_dim;
    p.in_dim  = in_dim;
    return p;
}

Stage4bEngine::Stage4bEngine(const std::string& bin_path, Stage4bConfig cfg)
    : cfg_(cfg), weights_(bin_path),
      off_(plan_pool(cfg)),
      pool_(off_.total_bytes) {
    input_proj_w_ = (const float*) weights_.get("input_proj.weight").data;
    input_proj_b_ = (const float*) weights_.get("input_proj.bias").data;
    pos_          = (const float*) weights_.get("pos").data;
    head_w_       = (const float*) weights_.get("head.weight").data;
    head_b_       = (const float*) weights_.get("head.bias").data;

    blocks_.resize(cfg_.n_layers);
    const int D = cfg_.d_model, F = cfg_.d_ff;
    for (int L = 0; L < cfg_.n_layers; L++) {
        std::string p = "blocks." + std::to_string(L) + ".";
        BlockPtrs& b = blocks_[L];
        b.ln1_w     = (const float*) weights_.get(p+"ln1.weight").data;
        b.ln1_b     = (const float*) weights_.get(p+"ln1.bias").data;
        b.qkv       = make_q(weights_, p+"qkv",      3*D, D);
        b.attn_out  = make_q(weights_, p+"attn_out", D,   D);
        b.ln2_w     = (const float*) weights_.get(p+"ln2.weight").data;
        b.ln2_b     = (const float*) weights_.get(p+"ln2.bias").data;
        b.ff1       = make_q(weights_, p+"ff1",      F,   D);
        b.ff2       = make_q(weights_, p+"ff2",      D,   F);
    }
}

void Stage4bEngine::forward(const float* x, float* out) {
    const int T = cfg_.seq_len, D = cfg_.d_model, H = cfg_.n_heads, Dh = D / H, F = cfg_.d_ff;

    float* h      = pool_.fview(off_.h);
    float* qkv    = pool_.fview(off_.qkv);
    float* q      = pool_.fview(off_.q);
    float* k      = pool_.fview(off_.k);
    float* v      = pool_.fview(off_.v);
    float* scores = pool_.fview(off_.scores);
    float* attn   = pool_.fview(off_.attn);
    float* tmp    = pool_.fview(off_.tmp);
    float* ff     = pool_.fview(off_.ff);
    float* pooled = pool_.fview(off_.pooled);
    uint8_t* xq   = pool_.uview(off_.xq);

    linear_fp32(x, input_proj_w_, input_proj_b_, h, T, cfg_.n_vars, D);
    for (int i = 0; i < T*D; i++) h[i] += pos_[i];

    for (int L = 0; L < cfg_.n_layers; L++) {
        const BlockPtrs& bp = blocks_[L];

        layer_norm(h, bp.ln1_w, bp.ln1_b, tmp, T, D);
        quantized_linear_4b(tmp, bp.qkv, qkv, T, xq);

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

        quantized_linear_4b(attn, bp.attn_out, tmp, T, xq);
        for (int i = 0; i < T*D; i++) h[i] += tmp[i];

        layer_norm(h, bp.ln2_w, bp.ln2_b, tmp, T, D);
        quantized_linear_4b(tmp, bp.ff1, ff, T, xq);
        for (int i = 0; i < T*F; i++) ff[i] = gelu_tanh(ff[i]);
        quantized_linear_4b(ff, bp.ff2, tmp, T, xq);
        for (int i = 0; i < T*D; i++) h[i] += tmp[i];
    }

    std::memset(pooled, 0, D*sizeof(float));
    for (int t = 0; t < T; t++) for (int d = 0; d < D; d++) pooled[d] += h[t*D + d];
    for (int d = 0; d < D; d++) pooled[d] /= T;

    linear_fp32(pooled, head_w_, head_b_, out, 1, D, cfg_.horizon * cfg_.n_vars);
}
