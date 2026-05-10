#pragma once
#include "tensor.h"
#include <string>
#include <unordered_map>

class WeightLoader {
public:
    explicit WeightLoader(const std::string& bin_path);
    ~WeightLoader();

    const Tensor& get(const std::string& name) const;
    bool has(const std::string& name) const { return tensors_.count(name) > 0; }
    size_t size() const { return tensors_.size(); }
    const std::unordered_map<std::string, Tensor>& all() const { return tensors_; }

private:
    void*  base_ = nullptr;
    size_t file_size_ = 0;
    int    fd_ = -1;
    std::unordered_map<std::string, Tensor> tensors_;
};
