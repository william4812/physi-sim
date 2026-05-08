# physi-sim/src/core/CMakeLists.txt

add_library(core_logic STATIC
    Grid2D.cpp
)

# Use the standardized variable for clarity
target_include_directories(core_logic PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Crucial for Linux/Ubuntu: Link the filesystem library
target_link_libraries(core_logic PUBLIC stdc++fs)
