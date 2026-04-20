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

# 2. Project Library 
add_library(physi_core
    src/thermal/LinearDummySolver.cpp
    src/io/VTKWriter.cpp
)
target_include_directories(physi_core PUBLIC 
    #${CMAKE_CURRENT_SOURCE_DIR}/include/thermal
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 3. Main Executable (Demonstration)
add_executable(physi_sim src/main.cpp)
target_link_libraries(physi_sim PRIVATE physi_core)

# 4. Unit Test Executable
enable_testing()
# Group all test source files here
set(TEST_SOURCES
    tests/test_solver.cpp
    tests/test_io.cpp
)

add_executable(physi_tests ${TEST_SOURCES})
target_link_libraries(physi_tests PRIVATE 
    physi_core 
    gtest_main
)
include(GoogleTest)
gtest_discover_tests(physi_tests)
#add_test(NAME SolverLogicTest COMMAND physi_tests)
#add_test(NAME DataIntegrityTest COMMAND physi_tests)
#add_test(NAME LinearDummySolverTest COMMAND physi_tests)
