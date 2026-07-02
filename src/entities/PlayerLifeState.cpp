#include "sim/entities/Player.hpp"

namespace sim::entities {

const char* ToString(const PlayerLifeState state)
{
    switch (state) {
    case PlayerLifeState::Alive: return "Alive";
    case PlayerLifeState::Dead:   return "Dead";
    }
    return "Unknown";
}

} // namespace sim::entities
