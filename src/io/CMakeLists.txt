# physi-sim/src/io/CMakeLists.txt

add_library(io_component STATIC
    TelemetryWriter.cpp
    VTKWriter.cpp
)

# Use the standardized variable for clarity
target_include_directories(io_component PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Crucial for Linux/Ubuntu: Link the filesystem library
target_link_libraries(io_component PUBLIC stdc++fs)
