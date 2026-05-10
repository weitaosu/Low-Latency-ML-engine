#pragma once
#include <functional>
#include <string>

struct MeasureConfig {
    int    isolated_core   = 3;
    int    warmup_iters    = 1000;
    int    measure_iters   = 100000;
    std::string stage      = "unknown";
    std::string size_      = "small";
    std::string precision  = "fp32";
    std::string csv_path   = "";
};

void measure(std::function<void()> fn, const MeasureConfig& cfg);
