cmake_minimum_required(VERSION 3.16)
project(low_latency_ml CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Strong optimization, AVX2, FMA. Per-stage targets can override.
add_compile_options(-O3 -march=native -mtune=native -fno-omit-frame-pointer -Wall -Wextra)

add_library(common STATIC
    common/weight_loader.cpp
    common/npy_loader.cpp
    common/correctness.cpp
    common/measure.cpp
    common/alloc_guard.cpp
)
target_include_directories(common PUBLIC common)
find_package(Threads REQUIRED)
target_link_libraries(common PUBLIC Threads::Threads)

# Each stage adds its own subdirectory (uncomment as you build them)
add_subdirectory(stage0_naive)
add_subdirectory(stage1_prealloc)
add_subdirectory(stage2_fused)
add_subdirectory(stage3_tiled)
add_subdirectory(stage4a_int8_pertensor)
add_subdirectory(stage4b_int8_perchannel)
add_subdirectory(stage5_fp32_simd)
# add_subdirectory(stage5_int8_simd)   # left for follow-up session — needs careful intrinsics work
