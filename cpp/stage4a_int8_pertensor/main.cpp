#include "engine.h"
#include "correctness.h"
#include "measure.h"
#include "alloc_guard.h"
#include <cstdio>
#include <cstdlib>
#include <string>

static Stage4aConfig cfg_for(const std::string& size) {
    Stage4aConfig c;
    if (size == "medium") {
        c.d_model = 192; c.n_heads = 6; c.n_layers = 4; c.d_ff = 768;
    }
    return c;
}

int main(int argc, char** argv) {
    std::string size = (argc > 1) ? argv[1] : "small";
    const char* env = std::getenv("REPO");
    std::string repo = env ? env : ".";
    std::string bin   = repo + "/models/" + size + "_int8_4a.bin";
    std::string in_p  = repo + "/data/inputs_1k.npy";
    std::string out_p = repo + "/data/outputs_1k_" + size + ".npy";
    std::string csv   = repo + "/results/csvs/stage4a_" + size + "_int8_pertensor.csv";

    Stage4aConfig cfg = cfg_for(size);
    Stage4aEngine engine(bin, cfg);

    // INT8 correctness tolerance: looser than FP32 stages — quantization noise.
    auto res = check_correctness(in_p, out_p, cfg.seq_len, cfg.n_vars,
        [&](const float* x, float* y){ engine.forward(x, y); }, 5e-2);
    if (!res.passed) { fprintf(stderr, "correctness FAILED (INT8 tol=5e-2)\n"); return 1; }

    {
        std::vector<float> di(cfg.seq_len * cfg.n_vars, 0.f);
        std::vector<float> dout(cfg.horizon * cfg.n_vars, 0.f);
        AllocGuard g;
        engine.forward(di.data(), dout.data());
        g.assert_zero();
    }
    fprintf(stderr, "alloc guard: zero allocations on hot path PASS\n");

    const char* mi = std::getenv("MEASURE_ITERS");
    const char* wi = std::getenv("WARMUP_ITERS");
    std::vector<float> dummy_in(cfg.seq_len * cfg.n_vars, 0.f);
    std::vector<float> dummy_out(cfg.horizon * cfg.n_vars, 0.f);
    auto fn = [&](){ engine.forward(dummy_in.data(), dummy_out.data()); };

    MeasureConfig mc;
    mc.stage = "stage4a"; mc.size_ = size; mc.precision = "int8_pertensor";
    mc.warmup_iters  = wi ? std::atoi(wi) : 100;
    mc.measure_iters = mi ? std::atoi(mi) : 100000;
    mc.csv_path = csv;
    measure(fn, mc);
    return 0;
}
