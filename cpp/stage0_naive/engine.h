#pragma once
#include "weight_loader.h"

struct Stage0Config {
    int n_vars   = 7;
    int seq_len  = 96;
    int horizon  = 96;
    int d_model  = 128;     // 192 for medium
    int n_heads  = 4;       // 6 for medium
    int n_layers = 2;       // 4 for medium
    int d_ff     = 256;     // 768 for medium
};

class Stage0Engine {
public:
    Stage0Engine(const std::string& bin_path, Stage0Config cfg);
    void forward(const float* x, float* out);   // x: [seq_len*n_vars], out: [horizon*n_vars]
private:
    Stage0Config cfg_;
    WeightLoader weights_;
};
