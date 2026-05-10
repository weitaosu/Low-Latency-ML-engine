#pragma once
#include <atomic>

namespace ag {
    extern std::atomic<int64_t> alloc_count;
    extern std::atomic<int64_t> free_count;
    extern bool tracking;
}

class AllocGuard {
public:
    AllocGuard();
    ~AllocGuard();
    void assert_zero() const;
private:
    int64_t start_alloc_, start_free_;
};
