# src/apps/CMakeLists.txt

add_executable(export_chip_stack export_chip_stack.cpp)

# Link the required components. 
# Because thermal3d and io_component expose their include directories as PUBLIC,
# export_chip_stack automatically inherits them without extra boilerplate.
target_link_libraries(export_chip_stack 
    PRIVATE 
        thermal3d 
        io_component
)

# Optional: Set output directory for binaries if you like them centralized
set_target_properties(export_chip_stack PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
