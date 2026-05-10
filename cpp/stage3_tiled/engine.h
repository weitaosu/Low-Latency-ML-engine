#pragma once
#include "weight_loader.h"
#include <vector>
#include <cstdint>

struct Stage3Config {
    int n_vars   = 7;
    int seq_len  = 96;
    int horizon  = 96;
    int d_model  = 128;
    int n_heads  = 4;
    int n_layers = 2;
    int d_ff     = 256;
};

// Single contiguous activation pool — same as stage 1.
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

// Each large weight matrix is pre-formatted into blocked layout at construction.
// Layout: BlockedWeight stores W in tiles of [tile_n × tile_k] so the inner
// kernel reads contiguous bytes. We keep things simple: tile_n = TILE_N fixed,
// tile_k = K (the whole inner dim) — i.e. weights are laid out row-blocked by N.
//
// For a Linear with W shape [N, K] (PyTorch convention: out_features, in_features),
// blocked[block_n] holds W[block_n*TILE_N : (block_n+1)*TILE_N, :].
struct BlockedWeight {
    std::vector<float> data;   // size N_padded * K, row-major in tiles
    int N_padded;              // rounded up to multiple of TILE_N
    int N;                     // original
    int K;
};

struct BlockPtrs {
    const float* ln1_w; const float* ln1_b;
    const BlockedWeight* qkv_W; const float* qkv_b;
    const BlockedWeight* attn_out_W; const float* attn_out_b;
    const float* ln2_w; const float* ln2_b;
    const BlockedWeight* ff1_W; const float* ff1_b;
    const BlockedWeight* ff2_W; const float* ff2_b;
};

class Stage3Engine {
public:
    Stage3Engine(const std::string& bin_path, Stage3Config cfg);
    void forward(const float* x, float* out);
private:
    Stage3Config cfg_;
    WeightLoader weights_;
    PoolOffsets  off_;
    ActivationPool pool_;

    // Cached top-level pointers
    const float* input_proj_w_;
    const float* input_proj_b_;
    const float* pos_;
    const float* head_w_;
    const float* head_b_;

    // Pre-blocked weights — owned by the engine
    std::vector<BlockedWeight> blocked_storage_;
    std::vector<BlockPtrs> blocks_;
};
