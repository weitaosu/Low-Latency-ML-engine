#pragma once
#include "weight_loader.h"
#include <vector>
#include <cstdint>

struct Stage4aConfig {
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
    float*    fview(size_t off) { return reinterpret_cast<float*>(buf_.data() + off); }
    uint8_t*  uview(size_t off) { return buf_.data() + off; }
private:
    std::vector<uint8_t> buf_;
};

struct PoolOffsets {
    size_t h, qkv, q, k, v, scores, attn, tmp, ff, pooled;
    size_t xq;        // uint8 scratch for quantized activations
    size_t total_bytes;
};

// Per-Linear quantization parameters
struct QLinearParams {
    const int8_t*  W_q;       // [out, in]
    const int32_t* sum_W;     // [out]
    const float*   bias;      // [out] fp32 (post-dequant add)
    float a_scale;
    int   a_zp;
    float w_scale;            // per-tensor scalar
    int   out_dim;
    int   in_dim;
};

struct BlockPtrs {
    const float* ln1_w; const float* ln1_b;
    QLinearParams qkv;
    QLinearParams attn_out;
    const float* ln2_w; const float* ln2_b;
    QLinearParams ff1;
    QLinearParams ff2;
};

class Stage4aEngine {
public:
    Stage4aEngine(const std::string& bin_path, Stage4aConfig cfg);
    void forward(const float* x, float* out);
private:
    Stage4aConfig cfg_;
    WeightLoader weights_;
    PoolOffsets  off_;
    ActivationPool pool_;
    const float *input_proj_w_, *input_proj_b_, *pos_, *head_w_, *head_b_;
    std::vector<BlockPtrs> blocks_;
};
