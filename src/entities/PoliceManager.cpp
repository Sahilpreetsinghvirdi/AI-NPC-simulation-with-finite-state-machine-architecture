#include "sim/entities/PoliceManager.hpp"

#include <algorithm>
#include <cmath>

namespace sim::entities {

namespace {

constexpr float kSpawnRadius = 35.0f;
constexpr float kSpawnAngleStepRadians = 1.2566371f;

int ClampWantedLevel(const int wantedLevel)
{
    return std::clamp(wantedLevel, 0, Player::kMaxWantedLevel);
}

} // namespace

void PoliceManager::SyncToWantedLevel(const int wantedLevel, const sim::math::Vec2 playerPosition)
{
    const std::size_t targetCount = static_cast<std::size_t>(ClampWantedLevel(wantedLevel));

    while (policeNpcs_.size() > targetCount) {
        policeNpcs_.pop_back();
    }

    while (policeNpcs_.size() < targetCount) {
        policeNpcs_.emplace_back(CalculateSpawnPosition(playerPosition, policeNpcs_.size()));
    }
}

void PoliceManager::UpdateAll(const float deltaTime, Player& player)
{
    for (PoliceNpc& police : policeNpcs_) {
        police.Update(deltaTime, player);
    }
}

std::size_t PoliceManager::GetActiveCount() const
{
    return policeNpcs_.size();
}

bool PoliceManager::HasActivePolice() const
{
    return !policeNpcs_.empty();
}

bool PoliceManager::AnyInState(const NpcState state) const
{
    return std::any_of(policeNpcs_.begin(), policeNpcs_.end(), [state](const PoliceNpc& police) {
        return police.IsAlive() && police.GetState() == state;
    });
}

const std::vector<PoliceNpc>& PoliceManager::GetPoliceNpcs() const
{
    return policeNpcs_;
}

const PoliceNpc* PoliceManager::GetPrimaryPolice() const
{
    return policeNpcs_.empty() ? nullptr : &policeNpcs_.front();
}

const PoliceNpc* PoliceManager::GetNearestPolice(const sim::math::Vec2 position) const
{
    const PoliceNpc* nearestPolice = nullptr;
    float nearestDistanceSquared = 0.0f;

    for (const PoliceNpc& police : policeNpcs_) {
        if (!police.IsAlive()) {
            continue;
        }

        const float distanceSquared = sim::math::DistanceSquared(position, police.GetPosition());
        if (nearestPolice == nullptr || distanceSquared < nearestDistanceSquared) {
            nearestPolice = &police;
            nearestDistanceSquared = distanceSquared;
        }
    }

    return nearestPolice;
}

PoliceNpc* PoliceManager::GetNearestPolice(const sim::math::Vec2 position)
{
    PoliceNpc* nearestPolice = nullptr;
    float nearestDistanceSquared = 0.0f;

    for (PoliceNpc& police : policeNpcs_) {
        if (!police.IsAlive()) {
            continue;
        }

        const float distanceSquared = sim::math::DistanceSquared(position, police.GetPosition());
        if (nearestPolice == nullptr || distanceSquared < nearestDistanceSquared) {
            nearestPolice = &police;
            nearestDistanceSquared = distanceSquared;
        }
    }

    return nearestPolice;
}

sim::math::Vec2 PoliceManager::CalculateSpawnPosition(const sim::math::Vec2 playerPosition, const std::size_t index) const
{
    const float angle = static_cast<float>(index) * kSpawnAngleStepRadians;
    return {
        playerPosition.x + (std::cos(angle) * kSpawnRadius),
        playerPosition.y + (std::sin(angle) * kSpawnRadius)
    };
}

} // namespace sim::entities
