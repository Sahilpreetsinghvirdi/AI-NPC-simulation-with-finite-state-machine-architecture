add_executable(ai_npc_tests
    NpcFiniteStateMachineTests.cpp
    ../src/ai/NpcFiniteStateMachine.cpp
    ../src/entities/NpcState.cpp
    ../src/entities/NpcAction.cpp
    ../src/entities/Player.cpp
    ../src/entities/PlayerLifeState.cpp
    ../src/entities/PoliceNpc.cpp
    ../src/math/Vec2.cpp
    ../src/ai/Observation.cpp
)

target_include_directories(ai_npc_tests PRIVATE "${PROJECT_SOURCE_DIR}/include")

if(WIN32)
    target_compile_definitions(ai_npc_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

add_test(NAME ai_npc_fsm_tests COMMAND ai_npc_tests)
