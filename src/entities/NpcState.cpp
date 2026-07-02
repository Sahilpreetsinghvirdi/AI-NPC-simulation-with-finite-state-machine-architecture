#include "sim/entities/NpcState.hpp"

namespace sim::entities {

const char* ToString(const NpcState state)
{
    switch (state) {
    case NpcState::Idle:        return "Idle";
    case NpcState::Patrol:      return "Patrol";
    case NpcState::Pursue:      return "Pursue";
    case NpcState::Investigate: return "Investigate";
    }
    return "Unknown";
}

} // namespace sim::entities
