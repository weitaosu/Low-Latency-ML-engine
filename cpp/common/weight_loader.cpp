#include "weight_loader.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <cstdio>

static const char MAGIC[4] = {'L','L','M','L'};

WeightLoader::WeightLoader(const std::string& path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open " + path);
    struct stat st; fstat(fd_, &st);
    file_size_ = st.st_size;
    base_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (base_ == MAP_FAILED) throw std::runtime_error("mmap failed");

    const uint8_t* p = (const uint8_t*)base_;
    if (memcmp(p, MAGIC, 4) != 0) throw std::runtime_error("bad magic");
    p += 4;
    uint32_t version  = *(const uint32_t*)p; p += 4;
    uint32_t n_tensors = *(const uint32_t*)p; p += 4;
    if (version != 1) throw std::runtime_error("unsupported version");

    const uint8_t* end = (const uint8_t*)base_ + file_size_;
    for (uint32_t i = 0; i < n_tensors; i++) {
        uint16_t name_len = *(const uint16_t*)p; p += 2;
        std::string name((const char*)p, name_len); p += name_len;
        uint8_t dt = *p++; uint8_t rank = *p++;
        int64_t shape[4] = {0,0,0,0};
        for (int r = 0; r < rank; r++) { shape[r] = *(const int64_t*)p; p += 8; }

        // Pad to 32-byte alignment
        uintptr_t off = (uintptr_t)(p - (const uint8_t*)base_);
        uintptr_t pad = (-off) & 31;
        p += pad;

        Tensor t = make_view(const_cast<uint8_t*>(p), (DType)dt, rank, shape);
        tensors_[name] = t;
        p += t.nbytes();
        if (p > end) throw std::runtime_error("corrupted .bin (tensor " + name + ")");
    }
}

WeightLoader::~WeightLoader() {
    if (base_) munmap(base_, file_size_);
    if (fd_ >= 0) close(fd_);
}

const Tensor& WeightLoader::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("missing tensor: " + name);
    return it->second;
}
