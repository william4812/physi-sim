# physi-sim/src/io/CMakeLists.txt

add_library(io_component STATIC
    VTKWriter.cpp
)

# Link the interface headers
target_include_directories(io_component PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
