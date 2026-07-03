add_executable(ai_npc_tests
    NpcFiniteStateMachineTests.cpp
    ../src/Simulation.cpp
    ../src/core/Logger.cpp
    ../src/core/LogLevel.cpp
    ../src/core/SimulationTimer.cpp
    ../src/rl/EpisodeRecorder.cpp
    ../src/ai/IPolicy.cpp
    ../src/ai/FeedForwardNetwork.cpp
    ../src/ai/FsmPolicy.cpp
    ../src/ai/PersistentLearningPolicy.cpp
    ../src/ai/PpoOptimizer.cpp
    ../src/ai/PpoTrajectoryBuffer.cpp
    ../src/ai/NpcFiniteStateMachine.cpp
    ../src/ai/PolicyRegistry.cpp
    ../src/rl/CheckpointManager.cpp
    ../src/entities/NpcState.cpp
    ../src/entities/NpcAction.cpp
    ../src/entities/PoliceManager.cpp
    ../src/entities/Player.cpp
    ../src/entities/PlayerLifeState.cpp
    ../src/entities/PoliceNpc.cpp
    ../src/math/Vec2.cpp
    ../src/ai/Observation.cpp
)

target_include_directories(ai_npc_tests PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_compile_definitions(ai_npc_tests PRIVATE
    SIM_DATA_DIR="${SIM_DATA_DIR}"
    SIM_LOGS_DIR="${SIM_LOGS_DIR}"
    SIM_MODELS_DIR="${SIM_MODELS_DIR}"
)

if(WIN32)
    target_compile_definitions(ai_npc_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

add_test(NAME ai_npc_fsm_tests COMMAND ai_npc_tests)
