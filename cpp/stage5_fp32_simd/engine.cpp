// Stage 5 FP32 SIMD: AVX2 intrinsics in the GEMM microkernel.
//
// TILE_N = 16 (two __m256 lanes). Inner k loop does 2 FMAs per iter (one per
// lane), 16 outputs accumulated in parallel. For longer K we get good FMA
// throughput on Haswell+ (2 FMA units; 2 in-flight FMAs per iter saturates one).
//
// Matches the FP32 reduction order of stage 3 (no, slightly different — drift
// vs stage 3 is FP-reorder ~1e-5).
//
// GELU stays scalar tanh approximation — matches what the model was trained
// with (must not switch between exact/tanh).

#include "engine.h"
#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <stdexcept>

constexpr int TILE_N = 16;

static void linear_naive(const float* x, const float* W, const float* b,
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

// ---------- SIMD-friendly weight blocking ----------
// Source W: [N, K] row-major.
// Target:   [Np/TILE_N][K][TILE_N]  with zero padding on N.
//   blocked[bn*K*TILE_N + k*TILE_N + t]  =  W[bn*TILE_N + t, k]
static BlockedWeightSimd block_weight(const float* W, int N, int K) {
    BlockedWeightSimd bw;
    bw.N = N; bw.K = K;
    bw.N_padded = ((N + TILE_N - 1) / TILE_N) * TILE_N;
    bw.data.assign((size_t)bw.N_padded * K, 0.f);
    for (int bn = 0; bn < bw.N_padded; bn += TILE_N) {
        for (int t = 0; t < TILE_N; t++) {
            int n = bn + t;
            if (n >= N) continue;  // padding row stays 0
            for (int k = 0; k < K; k++) {
                bw.data[(size_t)bn * K + (size_t)k * TILE_N + t] = W[(size_t)n * K + k];
            }
        }
    }
    return bw;
}

// ---------- AVX2 tiled matmul ----------
// y[M, N] = x[M, K] @ W^T + b   where W is BlockedWeightSimd.
// Inner: per output tile (16 cols), accumulate over K with 2x __m256 FMAs.
static void linear_simd(const float* __restrict__ x,
                        const BlockedWeightSimd& W,
                        const float* __restrict__ b,
                        float* __restrict__ y,
                        int M) {
    const int K = W.K;
    const int N = W.N;
    const int Np = W.N_padded;
    const float* Wbase = W.data.data();
    for (int m = 0; m < M; m++) {
        const float* xm = x + (size_t)m * K;
        float* ym = y + (size_t)m * N;
        for (int bn = 0; bn < Np; bn += TILE_N) {
            __m256 acc_lo, acc_hi;
            if (b && bn + TILE_N <= N) {
                acc_lo = _mm256_loadu_ps(b + bn);
                acc_hi = _mm256_loadu_ps(b + bn + 8);
            } else {
                acc_lo = _mm256_setzero_ps();
                acc_hi = _mm256_setzero_ps();
                if (b) {
                    // Partial bias load for the last tile if N not multiple of 16
                    float tmp[TILE_N] = {0};
                    for (int t = 0; t < TILE_N && (bn + t) < N; t++) tmp[t] = b[bn + t];
                    acc_lo = _mm256_loadu_ps(tmp);
                    acc_hi = _mm256_loadu_ps(tmp + 8);
                }
            }
            const float* Wblk = Wbase + (size_t)bn * K;
            for (int k = 0; k < K; k++) {
                __m256 xk = _mm256_set1_ps(xm[k]);
                __m256 wlo = _mm256_loadu_ps(Wblk + (size_t)k * TILE_N);
                __m256 whi = _mm256_loadu_ps(Wblk + (size_t)k * TILE_N + 8);
                acc_lo = _mm256_fmadd_ps(xk, wlo, acc_lo);
                acc_hi = _mm256_fmadd_ps(xk, whi, acc_hi);
            }
            // Store, masking off padding
            int tmax = (bn + TILE_N <= N) ? TILE_N : (N - bn);
            if (tmax == TILE_N) {
                _mm256_storeu_ps(ym + bn,     acc_lo);
                _mm256_storeu_ps(ym + bn + 8, acc_hi);
            } else {
                float tmp[TILE_N];
                _mm256_storeu_ps(tmp,     acc_lo);
                _mm256_storeu_ps(tmp + 8, acc_hi);
                for (int t = 0; t < tmax; t++) ym[bn + t] = tmp[t];
            }
        }
    }
}

// ---------- pool ----------

static size_t aligned(size_t x, size_t a = 64) { return (x + a - 1) & ~(a - 1); }

static PoolOffsets plan_pool(const Stage5FConfig& cfg) {
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

Stage5FEngine::Stage5FEngine(const std::string& bin_path, Stage5FConfig cfg)
    : cfg_(cfg), weights_(bin_path),
      off_(plan_pool(cfg)),
      pool_(off_.total_bytes) {
    input_proj_w_ = (const float*) weights_.get("input_proj.weight").data;
    input_proj_b_ = (const float*) weights_.get("input_proj.bias").data;
    pos_          = (const float*) weights_.get("pos").data;
    head_w_       = (const float*) weights_.get("head.weight").data;
    head_b_       = (const float*) weights_.get("head.bias").data;

    blocked_storage_.reserve(4 * cfg_.n_layers);
    blocks_.resize(cfg_.n_layers);
    const int D = cfg_.d_model, F = cfg_.d_ff;
    for (int L = 0; L < cfg_.n_layers; L++) {
        std::string p = "blocks." + std::to_string(L) + ".";
        blocked_storage_.push_back(block_weight((const float*)weights_.get(p+"qkv.weight").data, 3*D, D));
        const BlockedWeightSimd* qkv_W = &blocked_storage_.back();
        blocked_storage_.push_back(block_weight((const float*)weights_.get(p+"attn_out.weight").data, D, D));
        const BlockedWeightSimd* attn_out_W = &blocked_storage_.back();
        blocked_storage_.push_back(block_weight((const float*)weights_.get(p+"ff1.weight").data, F, D));
        const BlockedWeightSimd* ff1_W = &blocked_storage_.back();
        blocked_storage_.push_back(block_weight((const float*)weights_.get(p+"ff2.weight").data, D, F));
        const BlockedWeightSimd* ff2_W = &blocked_storage_.back();

        BlockPtrs& b = blocks_[L];
        b.ln1_w      = (const float*) weights_.get(p+"ln1.weight").data;
        b.ln1_b      = (const float*) weights_.get(p+"ln1.bias").data;
        b.qkv_W      = qkv_W;
        b.qkv_b      = (const float*) weights_.get(p+"qkv.bias").data;
        b.attn_out_W = attn_out_W;
        b.attn_out_b = (const float*) weights_.get(p+"attn_out.bias").data;
        b.ln2_w      = (const float*) weights_.get(p+"ln2.weight").data;
        b.ln2_b      = (const float*) weights_.get(p+"ln2.bias").data;
        b.ff1_W      = ff1_W;
        b.ff1_b      = (const float*) weights_.get(p+"ff1.bias").data;
        b.ff2_W      = ff2_W;
        b.ff2_b      = (const float*) weights_.get(p+"ff2.bias").data;
    }
}

void Stage5FEngine::forward(const float* x, float* out) {
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

    linear_naive(x, input_proj_w_, input_proj_b_, h, T, cfg_.n_vars, D);
    for (int i = 0; i < T*D; i++) h[i] += pos_[i];

    for (int L = 0; L < cfg_.n_layers; L++) {
        const BlockPtrs& bp = blocks_[L];

        layer_norm(h, bp.ln1_w, bp.ln1_b, tmp, T, D);
        linear_simd(tmp, *bp.qkv_W, bp.qkv_b, qkv, T);

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

        linear_simd(attn, *bp.attn_out_W, bp.attn_out_b, tmp, T);
        for (int i = 0; i < T*D; i++) h[i] += tmp[i];

        layer_norm(h, bp.ln2_w, bp.ln2_b, tmp, T, D);
        linear_simd(tmp, *bp.ff1_W, bp.ff1_b, ff, T);
        for (int i = 0; i < T*F; i++) ff[i] = gelu_tanh(ff[i]);
        linear_simd(ff, *bp.ff2_W, bp.ff2_b, tmp, T);
        for (int i = 0; i < T*D; i++) h[i] += tmp[i];
    }

    std::memset(pooled, 0, D*sizeof(float));
    for (int t = 0; t < T; t++) for (int d = 0; d < D; d++) pooled[d] += h[t*D + d];
    for (int d = 0; d < D; d++) pooled[d] /= T;

    linear_naive(pooled, head_w_, head_b_, out, 1, D, cfg_.horizon * cfg_.n_vars);
}
