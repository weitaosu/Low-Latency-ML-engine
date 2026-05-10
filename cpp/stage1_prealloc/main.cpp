#include "engine.h"
#include "correctness.h"
#include "measure.h"
#include "alloc_guard.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static Stage1Config cfg_for(const std::string& size) {
    Stage1Config c;
    if (size == "medium") {
        c.d_model = 192; c.n_heads = 6; c.n_layers = 4; c.d_ff = 768;
    }
    return c;
}

int main(int argc, char** argv) {
    std::string size = (argc > 1) ? argv[1] : "small";
    const char* env = std::getenv("REPO");
    std::string repo = env ? env : ".";
    std::string bin   = repo + "/models/" + size + ".bin";
    std::string in_p  = repo + "/data/inputs_1k.npy";
    std::string out_p = repo + "/data/outputs_1k_" + size + ".npy";
    std::string csv   = repo + "/results/csvs/stage1_" + size + "_fp32.csv";

    Stage1Config cfg = cfg_for(size);
    Stage1Engine engine(bin, cfg);

    // 1. Correctness — must pass at the same tolerance as stage 0
    auto res = check_correctness(in_p, out_p, cfg.seq_len, cfg.n_vars,
        [&](const float* x, float* y){ engine.forward(x, y); }, 1e-3);
    if (!res.passed) { fprintf(stderr, "correctness FAILED\n"); return 1; }

    // 2. Verify zero allocations on the hot path (key promise of stage 1)
    {
        std::vector<float> di(cfg.seq_len * cfg.n_vars, 0.f);
        std::vector<float> dout(cfg.horizon * cfg.n_vars, 0.f);
        AllocGuard g;
        engine.forward(di.data(), dout.data());
        g.assert_zero();   // aborts the program if forward() allocated
    }
    fprintf(stderr, "alloc guard: zero allocations on hot path PASS\n");

    // 3. Benchmark
    const char* mi = std::getenv("MEASURE_ITERS");
    const char* wi = std::getenv("WARMUP_ITERS");
    std::vector<float> dummy_in(cfg.seq_len * cfg.n_vars, 0.f);
    std::vector<float> dummy_out(cfg.horizon * cfg.n_vars, 0.f);
    auto fn = [&](){ engine.forward(dummy_in.data(), dummy_out.data()); };

    MeasureConfig mc;
    mc.stage = "stage1"; mc.size_ = size; mc.precision = "fp32";
    mc.warmup_iters  = wi ? std::atoi(wi) : 100;
    mc.measure_iters = mi ? std::atoi(mi) : 100000;
    mc.csv_path = csv;
    measure(fn, mc);
    return 0;
}
