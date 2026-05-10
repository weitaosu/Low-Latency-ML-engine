#include "npy_loader.h"
#include <cstdio>
#include <stdexcept>
#include <cstring>
#include <regex>

NpyArray load_npy(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);

    char magic[6];
    fread(magic, 1, 6, f);
    if (memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("not a .npy file");
    uint8_t major = 0, minor = 0;
    fread(&major, 1, 1, f); fread(&minor, 1, 1, f);

    uint32_t header_len = 0;
    if (major >= 2) { fread(&header_len, 4, 1, f); }
    else { uint16_t hl = 0; fread(&hl, 2, 1, f); header_len = hl; }

    std::string header(header_len, '\0');
    fread(&header[0], 1, header_len, f);

    if (header.find("'descr': '<f4'") == std::string::npos &&
        header.find("'descr': '|f4'") == std::string::npos)
        throw std::runtime_error("only float32 .npy supported (got: " + header.substr(0, 80) + ")");
    if (header.find("'fortran_order': False") == std::string::npos)
        throw std::runtime_error("only C-order supported");

    NpyArray out;
    std::regex re(R"('shape':\s*\(([^)]*)\))");
    std::smatch m;
    if (!std::regex_search(header, m, re)) throw std::runtime_error("shape parse failed");
    std::string shape_str = m[1];
    int64_t total = 1;
    for (size_t i = 0, j; i < shape_str.size(); i = j + 1) {
        j = shape_str.find(',', i);
        if (j == std::string::npos) j = shape_str.size();
        std::string tok = shape_str.substr(i, j - i);
        size_t k = 0; while (k < tok.size() && (tok[k] == ' ' || tok[k] == '\t')) k++;
        if (k < tok.size()) { int64_t v = std::stoll(tok.substr(k)); out.shape.push_back(v); total *= v; }
    }

    out.data.resize(total);
    if ((int64_t)fread(out.data.data(), 4, total, f) != total)
        throw std::runtime_error("short read");
    fclose(f);
    return out;
}
