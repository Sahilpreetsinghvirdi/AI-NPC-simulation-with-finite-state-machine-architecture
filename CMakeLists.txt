cmake_minimum_required(VERSION 3.20)

# Root project definition. All targets inherit C++20 and output paths from here.
project(AiNpcSimulation
    VERSION 0.1.0
    DESCRIPTION "AI-driven NPC simulation sandbox"
    LANGUAGES CXX
)

# Centralized compiler and platform settings (Windows / VS2022 friendly).
include(cmake/ProjectOptions.cmake)
include(cmake/FetchRaylib.cmake)

# Runtime folders referenced by future modules (logging, data loading, model I/O).
set(SIM_DATA_DIR "${CMAKE_SOURCE_DIR}/data" CACHE PATH "Simulation data directory")
set(SIM_LOGS_DIR "${CMAKE_SOURCE_DIR}/logs" CACHE PATH "Simulation log output directory")
set(SIM_MODELS_DIR "${CMAKE_SOURCE_DIR}/models" CACHE PATH "Future RL model artifacts directory")

# Core executable and optional test suite.
add_subdirectory(src)

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
