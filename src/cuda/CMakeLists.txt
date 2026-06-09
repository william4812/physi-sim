# src/cuda/CMakeLists.txt

add_library(cuda_solver STATIC
    CudaJacobiSolver.cu
    CudaTDMASolver.cu
)

target_include_directories(cuda_solver PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(cuda_solver
    PRIVATE core_logic
)

set_property(TARGET cuda_solver PROPERTY CUDA_ARCHITECTURES 75)
