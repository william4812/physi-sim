# Create the library for the 3D thermal physics package
#
# Conduction3D: 3D finite-volume conduction with Robin (convective) boundaries.
# This component grows to hold the material models k(T) (Phase 2), the chip-stack
# model (Phase 3), and the multiscale bridge + ScaleManager (Phase 4).
add_library(thermal3d STATIC
    Conduction3D.cpp
    # conduction_kernel.f90    <-- when a Fortran kernel lands here, add it AND
    #                              set LINKER_LANGUAGE CXX below (see 4th Principle)
)

# 1st Principle: Proper Header Propagation
# We go up two levels to find the project's 'include' directory.
# PUBLIC so consumers (physi_tests, and later the chip-stack demo) can write
# #include "thermal3d/Conduction3D.hpp" without repeating this path.
target_include_directories(thermal3d PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# 2nd Principle: Dependency Management — currently ZERO dependencies, by design.
# thermal3d links nothing: not core_logic (Grid2D is 2D), not io_component,
# not solver_component.
#
# The invariant that matters is NOT "never link anything" — it is:
#     thermal3d must never depend on the 2D core (Grid2D / ISolver),
#     and any dependency it does acquire must be PRIVATE and physics-driven.
#
# That is what keeps this component VERIFIABLE IN ISOLATION: the MMS
# order-of-accuracy test measures this physics and nothing else, and no change
# in the 2D stack can silently perturb it.
#   -> Phase 3 (VTK export of the chip stack):
#      target_link_libraries(thermal3d PRIVATE io_component)
#      PRIVATE, so the dependency does not leak to thermal3d's consumers.

# 3rd Principle: A component declares its own language requirement.
# C++17 is required (nested namespace physi_sim::thermal3d). Declaring it on the
# target means this component still builds correctly if the root default changes.
target_compile_features(thermal3d PUBLIC cxx_std_17)

# 4th Principle: Mixing languages is not the same as acquiring dependencies.
# A Fortran kernel compiled INTO this library is another translation unit inside
# the component — it adds no external coupling, so thermal3d stays zero-dependency
# and stays verifiable in isolation. MMS does not care what language the kernel is
# written in: it tests the NUMERICS. A C++ -> Fortran kernel swap must reproduce
# the same 2nd-order convergence, which makes the MMS test the regression net for
# exactly that swap.
#
# No set_target_properties(LINKER_LANGUAGE CXX) yet: thermal3d is currently pure
# C++, so CMake infers the linker language. The moment a .f90 is added above, the
# target holds BOTH C++ and Fortran objects and CMake can no longer infer which
# driver to link with — so uncomment the line below, exactly as thermal_backend
# does for diffusion_kernel.f90.
#
# set_target_properties(thermal3d PROPERTIES
#     LINKER_LANGUAGE CXX
# )
