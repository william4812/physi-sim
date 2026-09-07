# Create the library for the thermal package
add_library(thermal_backend STATIC
    diffusion_kernel.f90
    FortranBackend.cpp
    LinearDummySolver.cpp
)

# Propagate include paths transitively to dependent targets
target_include_directories(thermal_backend
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

# Link foundational core data structures and I/O handlers
target_link_libraries(thermal_backend
    PRIVATE
        core_logic
        io_component
)

# Enforce the C++ compiler/linker driver when combining C++ and Fortran sources
set_target_properties(thermal_backend PROPERTIES
    LINKER_LANGUAGE CXX
)

# Apply unified engineering profile (C++ & Fortran diagnostics, hardening, -march)
target_apply_engineering_defaults(thermal_backend)
