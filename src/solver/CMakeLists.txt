# src/solver/CMakeLists.txt

add_library(solver_component STATIC
    ConcurrentSolverRunner.cpp
    JacobiCPU.cpp
    ProfilingHarness.cpp
    SolverFactory.cpp
    SolverFSM.cpp
    TDMACPU.cpp
)

# Propagate include paths transitively to dependent targets
target_include_directories(solver_component
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

# Enforce strict public vs. private interface boundary
target_link_libraries(solver_component
    PUBLIC
        core_logic
    PRIVATE
        thermal_backend
)

# Apply unified engineering profile (warnings, hardening, optimization flags)
target_apply_engineering_defaults(solver_component)
