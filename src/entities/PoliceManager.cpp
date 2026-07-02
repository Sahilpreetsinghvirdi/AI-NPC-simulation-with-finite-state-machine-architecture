#include "sim/entities/PoliceManager.hpp"

#include <algorithm>
#include <cmath>

namespace sim::entities {

namespace {

constexpr float kSpawnRadius = 35.0f;
constexpr float kSpawnAngleStepRadians = 1.2566371f;
constexpr float kPredictionMinSeconds = 0.35f;
constexpr float kPredictionMaxSeconds = 1.4f;
constexpr float kCutoffDistance = 14.0f;
constexpr float kSeparationRadius = 12.0f;
constexpr float kSeparationStrength = 36.0f;
constexpr float kPlayerDamageInvulnerabilitySeconds = 0.85f;

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

void PoliceManager::UpdateAll(const float deltaTime, Player& player, const sim::math::Vec2 playerVelocity)
{
    playerDamageCooldown_ = std::max(0.0f, playerDamageCooldown_ - deltaTime);
    bool damageAppliedThisFrame = false;

    for (std::size_t index = 0; index < policeNpcs_.size(); ++index) {
        PoliceNpc& police = policeNpcs_[index];
        const sim::math::Vec2 target = CalculateInterceptTarget(police, player, playerVelocity, index);
        const sim::math::Vec2 separation = CalculateSeparationForce(index);
        const bool canDamagePlayer = playerDamageCooldown_ <= 0.0f && !damageAppliedThisFrame;

        if (police.Update(deltaTime, player, target, separation, canDamagePlayer)) {
            damageAppliedThisFrame = true;
            playerDamageCooldown_ = kPlayerDamageInvulnerabilitySeconds;
        }
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

sim::math::Vec2 PoliceManager::CalculateInterceptTarget(const PoliceNpc& police,
                                                        const Player& player,
                                                        const sim::math::Vec2 playerVelocity,
                                                        const std::size_t index) const
{
    const float distance = sim::math::Distance(police.GetPosition(), player.GetPosition());
    const float speed = std::max(police.GetCurrentSpeed(), PoliceNpc::kBaseSpeed);
    const float predictionSeconds = std::clamp(distance / speed * 0.45f,
                                               kPredictionMinSeconds,
                                               kPredictionMaxSeconds);
    const sim::math::Vec2 predictedPosition = player.GetPosition() + (playerVelocity * predictionSeconds);

    if (index == 0 || playerVelocity.LengthSquared() <= sim::math::Vec2::kEpsilon) {
        return predictedPosition;
    }

    const sim::math::Vec2 forward = playerVelocity.Normalized();
    const sim::math::Vec2 right{forward.y, -forward.x};
    const float sideSign = (index % 2 == 0) ? 1.0f : -1.0f;
    const float ring = 1.0f + static_cast<float>(index / 3);
    const float ahead = (index == 1 || index == 2) ? kCutoffDistance : kCutoffDistance * 0.45f;
    const float side = kCutoffDistance * sideSign * ring;

    return predictedPosition + (forward * ahead) + (right * side);
}

sim::math::Vec2 PoliceManager::CalculateSeparationForce(const std::size_t index) const
{
    if (index >= policeNpcs_.size()) {
        return {};
    }

    sim::math::Vec2 separation;
    const sim::math::Vec2 ownPosition = policeNpcs_[index].GetPosition();

    for (std::size_t otherIndex = 0; otherIndex < policeNpcs_.size(); ++otherIndex) {
        if (otherIndex == index) {
            continue;
        }

        const sim::math::Vec2 offset = ownPosition - policeNpcs_[otherIndex].GetPosition();
        const float distance = offset.Length();
        if (distance <= sim::math::Vec2::kEpsilon || distance >= kSeparationRadius) {
            continue;
        }

        const float push = (1.0f - (distance / kSeparationRadius)) * kSeparationStrength;
        separation += offset.Normalized() * push;
    }

    return separation;
}

} // namespace sim::entities
