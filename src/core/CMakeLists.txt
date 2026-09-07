# physi-sim/src/core/CMakeLists.txt

add_library(core_logic
    Grid2D.cpp
    solver.cpp
)

# Propagate include paths transitively to dependent targets
target_include_directories(core_logic
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

# Apply unified engineering defaults (warnings, optimization, security flags)
target_apply_engineering_defaults(core_logic)
