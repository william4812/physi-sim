# src/solver/CMakeLists.txt

# Define the component library
add_library(solver_component STATIC
    SolverFactory.cpp
    JacobiCPU.cpp
    TDMACPU.cpp
    ProfilingHarness.cpp
    SolverFSM.cpp
    ConcurrentSolverRunner.cpp
)

# Propagate include directories cleanly to consumers
target_include_directories(solver_component PUBLIC 
    ${CMAKE_SOURCE_DIR}/src
)
