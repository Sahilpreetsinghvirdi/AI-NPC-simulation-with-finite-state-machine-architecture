# Shared build configuration for all targets in this repository.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile_commands.json for tooling; harmless on MSVC.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Debug on single-config generators; VS multi-config ignores this.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# Place binaries under build/bin/<Config> for predictable VS/CMake workflows.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${CONFIG} CONFIG_UPPER)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_BINARY_DIR}/bin/${CONFIG}")
endforeach()

# Public headers live in include/; each module adds its own subdirectory as needed.
include_directories("${CMAKE_SOURCE_DIR}/include")

# Tests are opt-in so the sandbox executable builds quickly by default.
option(BUILD_TESTING "Build unit tests" OFF)

# Warnings treated as quality gates; /W4 is the MSVC equivalent of high warning level.
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()
