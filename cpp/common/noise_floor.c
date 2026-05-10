// Noise-floor sanity program. Build:
//   gcc -O2 -o noise_floor cpp/common/noise_floor.c
// Run pinned to the isolated core:
//   taskset -c 3 ./noise_floor > results/csvs/noise_floor.csv
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <sys/mman.h>

#define ITERS 1000000

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

int main(void) {
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        fprintf(stderr, "warning: mlockall failed (need CAP_IPC_LOCK)\n");
    }
    uint64_t *deltas = malloc(sizeof(uint64_t) * ITERS);
    if (!deltas) return 1;

    // Warmup
    for (int i = 0; i < 10000; i++) (void)now_ns();

    // Measure
    for (int i = 0; i < ITERS; i++) {
        uint64_t a = now_ns();
        uint64_t b = now_ns();
        deltas[i] = b - a;
    }

    // Emit CSV
    printf("iter,nanoseconds\n");
    for (int i = 0; i < ITERS; i++) {
        printf("%d,%lu\n", i, deltas[i]);
    }
    free(deltas);
    return 0;
}
