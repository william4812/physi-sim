cmake_minimum_required(VERSION 3.14)
project(PhysiSim LANGUAGES CXX Fortran)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Dependency Management
include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Speed up config")

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
add_subdirectory(src/thermal)
add_subdirectory(src/io)
add_subdirectory(src/core)

# 3. Main Executable
add_executable(physi_sim src/main.cpp)
# We link to the specific library created in the subdirectory
target_link_libraries(physi_sim PRIVATE 
    thermal_backend
    io_component
    core_logic
)

# 4. Unit Test Infrastructure
enable_testing()
add_executable(physi_tests 
    tests/test_solver.cpp
    tests/test_io.cpp
    tests/test_memory.cpp
    tests/test_physics.cpp
    tests/test_config.cpp
)

target_link_libraries(physi_tests PRIVATE 
    thermal_backend
    io_component 
    core_logic
    gtest_main
)
include(GoogleTest)
gtest_discover_tests(physi_tests)
