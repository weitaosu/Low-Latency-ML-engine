#pragma once
#include <string>
#include <functional>

struct CorrectnessResult { double max_abs_diff; double mean_abs_diff; bool passed; };

// fn: (input ptr, output ptr) -> void; both buffers shape [seq_len, n_vars] float32
CorrectnessResult check_correctness(const std::string& inputs_path,
                                    const std::string& outputs_ref_path,
                                    int seq_len, int n_vars,
                                    std::function<void(const float*, float*)> fn,
                                    double tolerance = 1e-4);
