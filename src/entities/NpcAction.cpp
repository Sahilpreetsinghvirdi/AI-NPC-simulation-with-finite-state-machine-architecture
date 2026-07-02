#include "sim/entities/NpcAction.hpp"

namespace sim::entities {

const char* ToString(const NpcAction action)
{
    switch (action) {
    case NpcAction::None:              return "None";
    case NpcAction::Wait:              return "Wait";
    case NpcAction::PatrolStep:        return "PatrolStep";
    case NpcAction::MoveTowardPlayer:  return "MoveTowardPlayer";
    case NpcAction::Attack:            return "Attack";
    }
    return "Unknown";
}

} // namespace sim::entities
