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

# 2. Modular Subdirectories
# This triggers the processing of src/thermal/CMakeLists.txt
add_subdirectory(src/thermal)
add_subdirectory(src/io)

# 3. Main Executable
add_executable(physi_sim src/main.cpp)
# We link to the specific library created in the subdirectory
target_link_libraries(physi_sim PRIVATE 
    thermal_backend
    io_component
)

# 4. Unit Test Infrastructure
enable_testing()
add_executable(physi_tests tests/test_solver.cpp)
target_link_libraries(physi_tests PRIVATE 
    thermal_backend
    io_component 
    gtest_main
)
include(GoogleTest)
gtest_discover_tests(physi_tests)
