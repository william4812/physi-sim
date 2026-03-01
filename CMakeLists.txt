cmake_minimum_required(VERSION 3.14)
project(PhysiSim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Dependency Management: Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0  # Use a specific, stable tag
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# 2. Project Library (Header-only for now)
add_library(physi_core INTERFACE)
target_include_directories(physi_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 3. Main Executable (Demonstration)
add_executable(physi_sim src/main.cpp)
target_link_libraries(physi_sim PRIVATE physi_core)

# 4. Unit Test Executable
enable_testing()
add_executable(physi_tests tests/test_solver.cpp)
target_link_libraries(physi_tests PRIVATE physi_core gtest_main)
add_test(NAME SolverLogicTest COMMAND physi_tests)
