cmake_minimum_required(VERSION 3.21)
project(PhysiSim LANGUAGES CXX Fortran)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 75)
    find_package(CUDAToolkit REQUIRED)
    set(PHYSI_SIM_CUDA_ENABLED ON)
    message(STATUS "CUDA found — building GPU solvers (sm_75)")
else()
    set(PHYSI_SIM_CUDA_ENABLED OFF)
    message(STATUS "CUDA not found — CPU-only build")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA default architecture — GTX 1650 is sm_75 (Turing).
# Setting this here means every CUDA target inherits it without repeating it.

# 1. Dependency Management
include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Speed up config")
set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/.deps)
set(FETCHCONTENT_QUIET OFF)  # shows download progress instead of hanging silently

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Fetch nlohmann_json for configuration and security data handling
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3 
)
FetchContent_MakeAvailable(json)

# 2. Modular Subdirectories
# This triggers the processing of src/thermal/CMakeLists.txt
include_directories(include)
add_subdirectory(src/thermal)
add_subdirectory(src/io)
add_subdirectory(src/core)
add_subdirectory(src/solver)

# 3. Main Executable
add_executable(physi_sim src/main.cpp)
# We link to the specific library created in the subdirectory
# src/cuda subdirectory:
if(PHYSI_SIM_CUDA_ENABLED)
    add_subdirectory(src/cuda)
    target_compile_definitions(solver_component PUBLIC PHYSI_SIM_CUDA_ENABLED)
endif()

# physi_sim executable:
target_link_libraries(physi_sim PRIVATE
    thermal_backend
    io_component
    core_logic
    solver_component
    $<$<BOOL:${PHYSI_SIM_CUDA_ENABLED}>:cuda_solver>
)

# 4. Unit Test Infrastructure
enable_testing()
add_executable(physi_tests
    # unit
    tests/unit/test_matrix_layout.cpp
    tests/unit/test_vtk_writer.cpp
    tests/unit/test_config_loader.cpp
    tests/unit/test_backend_dispatch.cpp
    tests/unit/test_solver_factory.cpp
    tests/unit/test_solver_fsm.cpp
    tests/unit/test_comparison_variants.cpp
    tests/unit/test_residual_history_csv.cpp
    tests/unit/test_residual_and_convergence.cpp

    # integration
    tests/integration/test_fortran_interop.cpp
    tests/integration/test_profiling_harness.cpp
    tests/integration/test_concurrent_runner.cpp
    
    # regression
    tests/regression/test_convergence_baselines.cpp
)

if(PHYSI_SIM_CUDA_ENABLED)
    target_sources(physi_tests PRIVATE
        tests/unit/test_cuda_jacobi_solver.cpp
    )
    target_compile_definitions(physi_tests PRIVATE PHYSI_SIM_CUDA_ENABLED)
endif()


# physi_tests executable:
target_link_libraries(physi_tests PRIVATE
    solver_component
    core_logic
    thermal_backend
    io_component
    $<$<BOOL:${PHYSI_SIM_CUDA_ENABLED}>:cuda_solver>
    $<$<BOOL:${PHYSI_SIM_CUDA_ENABLED}>:CUDA::cudart>
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(physi_tests)
