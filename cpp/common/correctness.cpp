#include "correctness.h"
#include "npy_loader.h"
#include <vector>
#include <cmath>
#include <cstdio>

CorrectnessResult check_correctness(const std::string& inputs_path,
                                    const std::string& outputs_ref_path,
                                    int seq_len, int n_vars,
                                    std::function<void(const float*, float*)> fn,
                                    double tolerance) {
    auto in  = load_npy(inputs_path);
    auto ref = load_npy(outputs_ref_path);
    int64_t per_sample = (int64_t)seq_len * n_vars;
    int64_t n = in.numel() / per_sample;

    std::vector<float> y(per_sample);
    double max_diff = 0.0, sum_diff = 0.0;
    int64_t total_elem = 0;

    for (int64_t i = 0; i < n; i++) {
        const float* x  = in.data.data()  + i * per_sample;
        const float* yr = ref.data.data() + i * per_sample;
        fn(x, y.data());
        for (int64_t k = 0; k < per_sample; k++) {
            double d = std::fabs(y[k] - yr[k]);
            if (d > max_diff) max_diff = d;
            sum_diff += d;
            total_elem++;
        }
    }
    CorrectnessResult r;
    r.max_abs_diff = max_diff;
    r.mean_abs_diff = sum_diff / total_elem;
    r.passed = max_diff < tolerance;
    fprintf(stderr, "correctness: max-abs-diff=%.2e mean-abs-diff=%.2e tol=%.0e %s\n",
            r.max_abs_diff, r.mean_abs_diff, tolerance, r.passed ? "PASS" : "FAIL");
    return r;
}
