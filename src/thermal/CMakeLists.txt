# Create the library for the thermal package
add_library(thermal_backend
    FortranBackend.cpp
    LinearDummySolver.cpp
    diffusion_kernel.f90
)

# 1st Principle: Proper Header Propagation
# We go up two levels to find the project's 'include' directory
target_include_directories(thermal_backend PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
)

# 2nd Principle: Dependency Management
# This tells the linker that Thermal needs IO.
# It also allows #include "io/..." because io_component exports that path PUBLICLY.
target_link_libraries(thermal_backend
    PRIVATE io_component
)

# Ensure the C++ and Fortran compilers agree on the ABI
set_target_properties(thermal_backend PROPERTIES
    LINKER_LANGUAGE CXX
)
