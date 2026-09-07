cmake_minimum_required(VERSION 3.21)
project(PhysiSim LANGUAGES CXX Fortran)

enable_testing()

# ==============================================================================
# 0. High-Level Engineering Feature Switches & Toolchain Auto-Detection
# ==============================================================================
# 1st Principle: Expose explicit feature toggles to decouple target behavior 
# from the host toolchain while maintaining reproducible build state across presets.
option(PHYSI_SIM_ENABLE_HARDENING  "Enable stack protection, fortify source, and relro" OFF)
option(PHYSI_SIM_ENABLE_ASSERTIONS "Enable STL container assertions in Debug"          ON)
set(PHYSI_SIM_MARCH "" CACHE STRING "Target CPU microarchitecture (e.g., native, x86-64-v3)")

# CUDA Toolchain Detection
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 75)
    find_package(CUDAToolkit REQUIRED)
    set(PHYSI_SIM_CUDA_ENABLED ON)
    message(STATUS "CUDA found — building GPU solvers (sm_75)")
else()
    set(PHYSI_SIM_CUDA_ENABLED OFF)
    message(STATUS "CUDA not found — CPU-only build")
endif()

# Enforce uniform C++ language boundaries across all internal targets
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ==============================================================================
# Target-Scoped Engineering Profile Applier
# ==============================================================================
# 1st Principle (Encapsulation): Apply diagnostics and flags strictly per-target 
# via generator expressions rather than setting global directory properties. 
# This prevents project-level warnings/flags from leaking into external dependencies.
function(target_apply_engineering_defaults target_name)
    # 1. Compiler Warnings & Frame Pointers (Language-Scoped)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror -Wpedantic -Wconversion -Wshadow -Wold-style-cast>
            $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4 /WX /permissive->
        )
    endif()

    # 2. Fortran Diagnostics & Bounds Checking
    if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:Fortran>:$<$<CONFIG:Debug>:-g -O0 -fno-omit-frame-pointer -fcheck=all -fbacktrace>>
        )
    endif()

    # 3. Microarchitecture Vectorization Tuning
    if(PHYSI_SIM_MARCH AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target_name} PRIVATE -march=${PHYSI_SIM_MARCH})
        if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
            target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-march=${PHYSI_SIM_MARCH}>)
        endif()
    endif()

    # 4. Defense-in-Depth Binary Hardening
    # 1st Principle: Mitigate stack buffer overflows, enforce memory segment 
    # read-only protections (RELRO), and enable ASLR/PIE at the binary layout level.
    if(PHYSI_SIM_ENABLE_HARDENING)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_definitions(${target_name} PRIVATE _FORTIFY_SOURCE=2)
            target_compile_options(${target_name} PRIVATE -fstack-protector-strong)
            if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
                target_link_options(${target_name} PRIVATE -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -pie)
            endif()
        endif()
    endif()

    # 5. Debug Assertion Invariant Enforcement
    if(PHYSI_SIM_ENABLE_ASSERTIONS)
        target_compile_definitions(${target_name} PRIVATE $<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)
    endif()

    # 6. CUDA GPU Kernel Debug Instrumentation
    if(PHYSI_SIM_CUDA_ENABLED)
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-g -G>>
        )
    endif()
endfunction()

# CUDA default architecture — GTX 1650 is sm_75 (Turing).
# Setting this here means every CUDA target inherits it without repeating it.

# ==============================================================================
# 1. Dependency Management (Isolated Third-Party Subbuilds)
# ==============================================================================
include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Speed up config")
set(FETCHCONTENT_QUIET OFF)

# 1st Principle (Build Tree Isolation): NEVER override FETCHCONTENT_BASE_DIR to a 
# shared directory like ${CMAKE_SOURCE_DIR}/.deps. Dependencies must build inside 
# each preset's unique binary directory (${CMAKE_BINARY_DIR}) to prevent generator 
# state pollution (e.g., Ninja vs Unix Makefiles cache collisions).

# MODERN CMAKE: Idempotent and policy-compliant
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    # Optional L6 hardening: enforce shallow clones for large HPC dependencies
    GIT_SHALLOW    TRUE
    EXCLUDE_FROM_ALL TRUE
)

# Encapsulates check, population, directory injection, and target exposure
FetchContent_MakeAvailable(googletest)

# 1st Principle (Warning Suppression): Mark external headers as SYSTEM headers 
# so compiler warning flags (-Werror) applied to internal code do not fail 3rd-party code.
foreach(dep gtest gtest_main gmock gmock_main)
    if(TARGET ${dep})
        set_target_properties(${dep} PROPERTIES
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:${dep},INTERFACE_INCLUDE_DIRECTORIES>
        )
    endif()
endforeach()

FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3 
)
FetchContent_MakeAvailable(json)
if(TARGET nlohmann_json)
    set_target_properties(nlohmann_json PROPERTIES
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>
    )
endif()

if(MSVC)
    set(gtest_force_shared_crt ON CACHE BOOL "Force MSVC CRT matching" FORCE)
endif()

# ==============================================================================
# 2. Modular Subdirectories & Target Profile Enforcement
# ==============================================================================
include_directories(include)

add_subdirectory(src/thermal)
target_apply_engineering_defaults(thermal_backend)

add_subdirectory(src/io)
target_apply_engineering_defaults(io_component)

add_subdirectory(src/core)
target_apply_engineering_defaults(core_logic)

add_subdirectory(src/thermal3d)
target_apply_engineering_defaults(thermal3d)

add_subdirectory(src/solver)
target_apply_engineering_defaults(solver_component)

add_subdirectory(src/apps)

# ==============================================================================
# 3. Main Application Executable
# ==============================================================================
add_executable(physi_sim src/main.cpp)
target_apply_engineering_defaults(physi_sim)

if(PHYSI_SIM_CUDA_ENABLED)
    add_subdirectory(src/cuda)
    target_compile_definitions(solver_component PUBLIC PHYSI_SIM_CUDA_ENABLED)
    if(TARGET cuda_solver)
        target_apply_engineering_defaults(cuda_solver)
    endif()
endif()

target_link_libraries(physi_sim PRIVATE
    thermal_backend
    io_component
    core_logic
    solver_component
    $<$<BOOL:${PHYSI_SIM_CUDA_ENABLED}>:cuda_solver>
)

add_executable(physi_tests
    tests/unit/test_matrix_layout.cpp
    tests/unit/test_vtk_writer.cpp
    tests/unit/test_config_loader.cpp
    tests/unit/test_backend_dispatch.cpp
    tests/unit/test_solver_factory.cpp
    tests/unit/test_solver_fsm.cpp
    tests/unit/test_comparison_variants.cpp
    tests/unit/test_residual_history_csv.cpp
    tests/unit/test_residual_and_convergence.cpp
    tests/unit/test_field_export_vti.cpp    
    tests/integration/test_fortran_interop.cpp
    tests/integration/test_profiling_harness.cpp
    tests/integration/test_concurrent_runner.cpp
    tests/regression/test_convergence_baselines.cpp
    tests/regression/test_solution_validation.cpp
    tests/verification/test_mms_conduction3d.cpp
    tests/verification/test_electro_thermal_physics.cpp
    tests/verification/test_thermal_conduction_physics.cpp
    tests/verification/test_anisotropic_domain.cpp
)
target_apply_engineering_defaults(physi_tests)

if(PHYSI_SIM_CUDA_ENABLED)
    target_sources(physi_tests PRIVATE
        tests/unit/test_cuda_jacobi_solver.cpp
        tests/unit/test_cuda_tdma_solver.cpp
    )
    target_compile_definitions(physi_tests PRIVATE PHYSI_SIM_CUDA_ENABLED)
endif()

target_link_libraries(physi_tests PRIVATE
    solver_component
    core_logic
    thermal_backend
    thermal3d 
    io_component
    $<$<BOOL:${PHYSI_SIM_CUDA_ENABLED}>:cuda_solver>
    $<$<BOOL:${PHYSI_SIM_CUDA_ENABLED}>:CUDA::cudart>
    gtest_main
)

# Platform-specific high-resolution clock linkage
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(physi_tests PRIVATE rt)
    target_link_libraries(core_logic PRIVATE rt)
endif()

include(GoogleTest)
gtest_discover_tests(physi_tests)
