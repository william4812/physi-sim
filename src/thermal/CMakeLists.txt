# Create the library for the thermal package
add_library(thermal_backend
    FortranBackend.cpp
    LinearDummySolver.cpp
    diffusion_kernel.f90
)

# 1st Principle: Proper Header Propagation
# We go up two levels to find the project's 'include' directory
target_include_directories(thermal_backend PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/
)

# Ensure the C++ and Fortran compilers agree on the ABI
set_target_properties(thermal_backend PROPERTIES
    LINKER_LANGUAGE CXX
)
