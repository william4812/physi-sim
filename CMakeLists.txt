cmake_minimum_required(VERSION 3.14)
project(PhysiSim LANGUAGES CXX Fortran)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
target_link_libraries(physi_sim PRIVATE 
    thermal_backend
    io_component
    core_logic
    solver_component
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

    # integration
    tests/integration/test_fortran_interop.cpp

    # regression
    tests/regression/test_convergence_baselines.cpp
)
 
target_link_libraries(physi_tests PRIVATE 
    solver_component
    core_logic
    thermal_backend
    io_component 
    gtest_main
)
include(GoogleTest)
gtest_discover_tests(physi_tests)
