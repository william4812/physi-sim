# src/cuda/CMakeLists.txt

# 1. GPU Solver Library (Target-Centric, No main() symbol)
add_library(cuda_solver STATIC
    CudaJacobiSolver.cu
    CudaTDMASolver.cu
)

target_include_directories(cuda_solver PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(cuda_solver PRIVATE
    core_logic
    CUDA::cudart
)

set_target_properties(cuda_solver PROPERTIES
    CUDA_ARCHITECTURES 75
    CUDA_SEPARABLE_COMPILATION ON
)

target_apply_engineering_defaults(cuda_solver)

# 2. Standalone GPU Utility Executable
add_executable(gpu_load gpu_load.cu)

target_compile_definitions(gpu_load PRIVATE GPU_LOAD_STANDALONE)

target_include_directories(gpu_load PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(gpu_load PRIVATE
    CUDA::cudart
)

set_target_properties(gpu_load PROPERTIES
    CUDA_ARCHITECTURES 75
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_apply_engineering_defaults(gpu_load)