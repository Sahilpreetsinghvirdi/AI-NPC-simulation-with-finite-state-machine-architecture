#include "sim/entities/PoliceManager.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

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
constexpr float kMeaningfulDistanceGain = 1.0f;
constexpr float kNoProgressStrategySeconds = 2.5f;

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
        lastPlayerDistances_.pop_back();
        noProgressTimers_.pop_back();
        strategyRevisions_.pop_back();
    }

    while (policeNpcs_.size() < targetCount) {
        policeNpcs_.emplace_back(CalculateSpawnPosition(playerPosition, policeNpcs_.size()));
        lastPlayerDistances_.push_back(std::numeric_limits<float>::max());
        noProgressTimers_.push_back(0.0f);
        strategyRevisions_.push_back(0);
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

        const float currentDistance = sim::math::Distance(police.GetPosition(), player.GetPosition());
        if (lastPlayerDistances_[index] - currentDistance >= kMeaningfulDistanceGain) {
            noProgressTimers_[index] = 0.0f;
            lastPlayerDistances_[index] = currentDistance;
        } else {
            noProgressTimers_[index] += deltaTime;
            if (noProgressTimers_[index] >= kNoProgressStrategySeconds) {
                ++strategyRevisions_[index];
                noProgressTimers_[index] = 0.0f;
                lastPlayerDistances_[index] = currentDistance;
            }
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

const char* PoliceManager::GetRoleName(const std::size_t index) const
{
    static constexpr const char* kRoles[] = {
        "Lead",
        "Left Flank",
        "Right Flank",
        "Cutoff",
        "Reserve"
    };

    return kRoles[index % 5];
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
    const int revision = (index < strategyRevisions_.size()) ? strategyRevisions_[index] : 0;
    const float revisionPredictionBias = 1.0f + (0.12f * static_cast<float>(revision % 3));
    const float predictionSeconds = std::clamp(distance / speed * 0.45f * revisionPredictionBias,
                                               kPredictionMinSeconds,
                                               kPredictionMaxSeconds);
    const sim::math::Vec2 predictedPosition = player.GetPosition() + (playerVelocity * predictionSeconds);

    sim::math::Vec2 forward = playerVelocity.Normalized();
    if (playerVelocity.LengthSquared() <= sim::math::Vec2::kEpsilon) {
        forward = (player.GetPosition() - police.GetPosition()).Normalized();
        if (forward.LengthSquared() <= sim::math::Vec2::kEpsilon) {
            forward = {1.0f, 0.0f};
        }
    }
    const sim::math::Vec2 right{forward.y, -forward.x};
    const float revisionSide = (revision % 2 == 0) ? 1.0f : -1.0f;

    float ahead = kCutoffDistance * 0.75f;
    float side = 0.0f;

    switch (index % 5) {
    case 0:
        ahead = kCutoffDistance * (1.05f + (0.15f * static_cast<float>(revision % 2)));
        side = (revision > 0) ? kCutoffDistance * 0.35f * revisionSide : 0.0f;
        break;
    case 1:
        ahead = kCutoffDistance * 1.15f;
        side = -kCutoffDistance * (1.0f + (0.2f * static_cast<float>(revision % 2)));
        break;
    case 2:
        ahead = kCutoffDistance * 1.15f;
        side = kCutoffDistance * (1.0f + (0.2f * static_cast<float>(revision % 2)));
        break;
    case 3:
        ahead = kCutoffDistance * (1.85f + (0.15f * static_cast<float>(revision % 3)));
        side = kCutoffDistance * 0.25f * revisionSide;
        break;
    case 4:
    default:
        ahead = kCutoffDistance * 0.35f;
        side = kCutoffDistance * 1.55f * revisionSide;
        break;
    }

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
