#pragma once

namespace sim::entities {

// Discrete action space placeholder for future reinforcement learning integration.
enum class NpcAction {
    None = 0,
    Wait,
    PatrolStep,
    MoveTowardPlayer,
    Attack
};

const char* ToString(NpcAction action);

} // namespace sim::entities
