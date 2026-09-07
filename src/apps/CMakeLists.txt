add_executable(export_chip_stack export_chip_stack.cpp)

# Explicit target include path
target_include_directories(export_chip_stack
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Explicit component linkage
target_link_libraries(export_chip_stack
    PRIVATE
        core_logic
        io_component
        solver_component
        thermal3d
)

# Centralize build artifacts to ${CMAKE_BINARY_DIR}/bin
set_target_properties(export_chip_stack PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Apply unified compiler warnings, hardening, and optimization flags
target_apply_engineering_defaults(export_chip_stack)
