#pragma once
#include <cstdint>
#include <cstddef>

enum DType : uint8_t { DT_FP32 = 0, DT_INT8 = 1, DT_INT32 = 2 };

inline size_t dtype_size(DType d) {
    switch (d) { case DT_FP32: return 4; case DT_INT8: return 1; case DT_INT32: return 4; }
    return 0;
}

struct Tensor {
    void*   data;
    int64_t shape[4];
    int64_t strides[4];        // elements, not bytes
    int     rank;
    DType   dtype;
    bool    owns_data;

    Tensor() : data(nullptr), rank(0), dtype(DT_FP32), owns_data(false) {
        for (int i = 0; i < 4; i++) shape[i] = 0, strides[i] = 0;
    }

    int64_t numel() const {
        int64_t n = 1; for (int i = 0; i < rank; i++) n *= shape[i]; return n;
    }
    size_t nbytes() const { return numel() * dtype_size(dtype); }
    float* fp32() const { return reinterpret_cast<float*>(data); }
    int8_t* i8() const  { return reinterpret_cast<int8_t*>(data); }
};

// Build row-major strides from shape
inline void compute_strides(Tensor& t) {
    int64_t s = 1;
    for (int i = t.rank - 1; i >= 0; i--) { t.strides[i] = s; s *= t.shape[i]; }
}

inline Tensor make_view(void* data, DType dt, int rank, const int64_t* shape) {
    Tensor t; t.data = data; t.dtype = dt; t.rank = rank; t.owns_data = false;
    for (int i = 0; i < rank; i++) t.shape[i] = shape[i];
    compute_strides(t);
    return t;
}
