#include "measure.h"
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#include <time.h>
#include <cstdio>
#include <vector>
#include <fstream>
#include <cstring>

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static void pin_to_core(int core) {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(core, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
}

static std::string read_first_line(const char* path) {
    std::ifstream f(path); std::string s; if (f) std::getline(f, s); return s;
}

void measure(std::function<void()> fn, const MeasureConfig& cfg) {
    pin_to_core(cfg.isolated_core);
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        fprintf(stderr, "warning: mlockall failed\n");
    }

    std::vector<uint64_t> timings(cfg.measure_iters);
    for (int i = 0; i < cfg.warmup_iters; i++) fn();
    for (int i = 0; i < cfg.measure_iters; i++) {
        uint64_t t0 = now_ns();
        fn();
        timings[i] = now_ns() - t0;
    }

    if (cfg.csv_path.empty()) return;
    FILE* f = fopen(cfg.csv_path.c_str(), "w");
    if (!f) { perror("fopen"); return; }

    // Self-documenting header (commented)
    fprintf(f, "# stage=%s size=%s precision=%s warmup=%d measure=%d core=%d\n",
            cfg.stage.c_str(), cfg.size_.c_str(), cfg.precision.c_str(),
            cfg.warmup_iters, cfg.measure_iters, cfg.isolated_core);
    fprintf(f, "# isolated_cpus=%s no_turbo=%s perf_paranoid=%s\n",
            read_first_line("/sys/devices/system/cpu/isolated").c_str(),
            read_first_line("/sys/devices/system/cpu/intel_pstate/no_turbo").c_str(),
            read_first_line("/proc/sys/kernel/perf_event_paranoid").c_str());
    fprintf(f, "stage,size,precision,iteration,nanoseconds\n");
    for (int i = 0; i < cfg.measure_iters; i++) {
        fprintf(f, "%s,%s,%s,%d,%lu\n",
                cfg.stage.c_str(), cfg.size_.c_str(), cfg.precision.c_str(),
                i, timings[i]);
    }
    fclose(f);
    fprintf(stderr, "wrote %s (%d samples)\n", cfg.csv_path.c_str(), cfg.measure_iters);
}
