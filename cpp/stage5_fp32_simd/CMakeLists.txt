add_executable(stage5_fp32_simd main.cpp engine.cpp)
target_link_libraries(stage5_fp32_simd PRIVATE common)
target_compile_options(stage5_fp32_simd PRIVATE -mavx2 -mfma)
