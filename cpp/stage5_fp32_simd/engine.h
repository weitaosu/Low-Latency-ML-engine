#pragma once
#include "weight_loader.h"
#include <vector>
#include <cstdint>

struct Stage5FConfig {
    int n_vars   = 7;
    int seq_len  = 96;
    int horizon  = 96;
    int d_model  = 128;
    int n_heads  = 4;
    int n_layers = 2;
    int d_ff     = 256;
};

class ActivationPool {
public:
    explicit ActivationPool(size_t bytes) : buf_(bytes) {}
    float* fview(size_t off) { return reinterpret_cast<float*>(buf_.data() + off); }
private:
    std::vector<uint8_t> buf_;
};

struct PoolOffsets {
    size_t h, qkv, q, k, v, scores, attn, tmp, ff, pooled;
    size_t total_bytes;
};

// SIMD-friendly blocked layout: [Np/TILE_N][K][TILE_N]
// For each output tile of TILE_N=16 columns, all 16 weights for k=0 are
// contiguous, then all 16 weights for k=1, etc. → __m256 loads work directly.
struct BlockedWeightSimd {
    std::vector<float> data;
    int N_padded;
    int N;
    int K;
};

struct BlockPtrs {
    const float* ln1_w; const float* ln1_b;
    const BlockedWeightSimd* qkv_W; const float* qkv_b;
    const BlockedWeightSimd* attn_out_W; const float* attn_out_b;
    const float* ln2_w; const float* ln2_b;
    const BlockedWeightSimd* ff1_W; const float* ff1_b;
    const BlockedWeightSimd* ff2_W; const float* ff2_b;
};

class Stage5FEngine {
public:
    Stage5FEngine(const std::string& bin_path, Stage5FConfig cfg);
    void forward(const float* x, float* out);
private:
    Stage5FConfig cfg_;
    WeightLoader weights_;
    PoolOffsets  off_;
    ActivationPool pool_;
    const float *input_proj_w_, *input_proj_b_, *pos_, *head_w_, *head_b_;
    std::vector<BlockedWeightSimd> blocked_storage_;
    std::vector<BlockPtrs> blocks_;
};
