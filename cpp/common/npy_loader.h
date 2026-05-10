#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct NpyArray {
    std::vector<int64_t> shape;
    std::vector<float>   data;   // float32 only
    int64_t numel() const { int64_t n=1; for (auto s : shape) n*=s; return n; }
};

NpyArray load_npy(const std::string& path);
