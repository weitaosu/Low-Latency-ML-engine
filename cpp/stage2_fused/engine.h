#pragma once
#include "weight_loader.h"
#include <vector>
#include <cstdint>

struct Stage2Config {
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
    float* view(size_t byte_offset) { return reinterpret_cast<float*>(buf_.data() + byte_offset); }
private:
    std::vector<uint8_t> buf_;
};

struct PoolOffsets {
    size_t h, qkv, q, k, v, scores, attn, tmp, ff, pooled;
    size_t total_bytes;
};

struct BlockPtrs {
    const float *ln1_w, *ln1_b;
    const float *qkv_w, *qkv_b;
    const float *attn_out_w, *attn_out_b;
    const float *ln2_w, *ln2_b;
    const float *ff1_w, *ff1_b;
    const float *ff2_w, *ff2_b;
};

class Stage2Engine {
public:
    Stage2Engine(const std::string& bin_path, Stage2Config cfg);
    void forward(const float* x, float* out);
private:
    Stage2Config cfg_;
    WeightLoader weights_;
    PoolOffsets  off_;
    ActivationPool pool_;
    const float *input_proj_w_, *input_proj_b_, *pos_, *head_w_, *head_b_;
    std::vector<BlockPtrs> blocks_;
};
