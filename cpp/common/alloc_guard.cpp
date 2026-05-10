#include "alloc_guard.h"
#include <cstdio>
#include <cstdlib>
#include <new>

namespace ag {
    std::atomic<int64_t> alloc_count{0};
    std::atomic<int64_t> free_count{0};
    bool tracking = false;
}

void* operator new(std::size_t n) {
    if (ag::tracking) ag::alloc_count.fetch_add(1, std::memory_order_relaxed);
    void* p = std::malloc(n);
    if (!p) throw std::bad_alloc();
    return p;
}
void operator delete(void* p) noexcept {
    if (ag::tracking) ag::free_count.fetch_add(1, std::memory_order_relaxed);
    std::free(p);
}
void operator delete(void* p, std::size_t) noexcept {
    if (ag::tracking) ag::free_count.fetch_add(1, std::memory_order_relaxed);
    std::free(p);
}

AllocGuard::AllocGuard() {
    start_alloc_ = ag::alloc_count.load();
    start_free_  = ag::free_count.load();
    ag::tracking = true;
}
AllocGuard::~AllocGuard() { ag::tracking = false; }

void AllocGuard::assert_zero() const {
    int64_t da = ag::alloc_count.load() - start_alloc_;
    int64_t df = ag::free_count.load()  - start_free_;
    if (da != 0 || df != 0) {
        std::fprintf(stderr, "AllocGuard: %ld allocs, %ld frees on hot path\n", da, df);
        std::abort();
    }
}
