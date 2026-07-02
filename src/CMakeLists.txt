# Main simulation executable. Additional source files will be appended per module.
add_executable(ai_npc_sim
    main.cpp
    Simulation.cpp
    core/LogLevel.cpp
    core/Logger.cpp
    core/SimulationTimer.cpp
    math/Vec2.cpp
    ai/Observation.cpp
    ai/NpcFiniteStateMachine.cpp
    entities/Player.cpp
    entities/PlayerLifeState.cpp
    entities/NpcState.cpp
    entities/NpcAction.cpp
    entities/PoliceNpc.cpp
    visualization/SimulationRenderer.cpp
)

target_link_libraries(ai_npc_sim PRIVATE raylib)

# Inject runtime directory paths so modules can locate data/logs without hard-coded paths.
target_compile_definitions(ai_npc_sim PRIVATE
    SIM_DATA_DIR="${SIM_DATA_DIR}"
    SIM_LOGS_DIR="${SIM_LOGS_DIR}"
    SIM_MODELS_DIR="${SIM_MODELS_DIR}"
)

# On Windows, ensure UTF-8 console output behaves consistently.
if(WIN32)
    target_compile_definitions(ai_npc_sim PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
