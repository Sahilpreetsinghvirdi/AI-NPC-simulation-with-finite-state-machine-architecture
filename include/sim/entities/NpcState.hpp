#pragma once

namespace sim::entities {

// High-level behavioral mode for NPC decision-making and observation encoding.
enum class NpcState {
    Idle = 0,
    Patrol,
    Pursue,
    Investigate
};

const char* ToString(NpcState state);

} // namespace sim::entities
