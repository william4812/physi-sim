# physi-sim/src/io/CMakeLists.txt

add_library(io_component STATIC
    ChipLayout.cpp
    ConfigLoader.cpp
    CSVWriter.cpp
    TelemetryWriter.cpp
    VTKWriter.cpp
)

# Propagate include paths transitively to dependent targets
target_include_directories(io_component
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

# Link JSON parsing target and foundational domain types
target_link_libraries(io_component
    PRIVATE
        nlohmann_json::nlohmann_json
        core_logic
)

# Apply unified engineering profile (warnings, hardening, optimization flags)
target_apply_engineering_defaults(io_component)
