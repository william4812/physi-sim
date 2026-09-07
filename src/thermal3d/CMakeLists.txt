add_library(thermal3d STATIC
    Conduction3D.cpp
    ElectroThermal3D.cpp
)

# Propagate include paths transitively to dependent targets
target_include_directories(thermal3d
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

# 1st Principle: Zero-Dependency Physics Isolation
# thermal3d explicitly avoids coupling to 2D core logic (Grid2D / ISolver) or I/O components.
# Decoupling physics solvers ensures they remain verifiable in isolation (e.g., Method of
# Manufactured Solutions / MMS spatial order-of-accuracy tests) without 2D stack pollution.

# Apply unified engineering profile (warnings, hardening, optimization flags)
target_apply_engineering_defaults(thermal3d)

# Reserved for mixed C++/Fortran linkage (e.g., when adding conduction_kernel.f90):
# set_target_properties(thermal3d PROPERTIES LINKER_LANGUAGE CXX)