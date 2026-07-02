#include "sim/Simulation.hpp"

#include "sim/entities/NpcState.hpp"

#include <algorithm>
#include <cmath>

namespace sim {

namespace {

constexpr float kPlayerBaseSpeed = 6.0f;
constexpr float kPlayerMaxSpeed = 18.0f;
constexpr float kPlayerAcceleration = 22.0f;
constexpr float kPlayerAccelerationDistance = 40.0f;
constexpr float kPlayerMaxTurnRateRadiansPerSecond = 2.6f;
constexpr float kPoliceDangerRadius = 70.0f;
constexpr float kPlayerPerceptionRadius = 24.0f;
constexpr float kObstacleMemoryDuration = 4.5f;
constexpr float kWorldMin = 6.0f;
constexpr float kWorldMax = 114.0f;
constexpr float kPlayerAttackRange = 8.0f;
constexpr float kPlayerAttackCooldown = 0.8f;
constexpr float kPlayerAttackDamage = 8.0f;
constexpr float kPursuitFailureDuration = 10.0f;
constexpr float kDeathPauseDuration = 5.0f;
constexpr int kRespawnWantedLevel = 3;

float MoveTowards(const float current, const float target, const float maxDelta)
{
    if (current < target) {
        return std::min(target, current + maxDelta);
    }
    return std::max(target, current - maxDelta);
}

float WrapAngle(const float radians)
{
    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kTwoPi = kPi * 2.0f;
    float result = std::fmod(radians + kPi, kTwoPi);
    if (result < 0.0f) {
        result += kTwoPi;
    }
    return result - kPi;
}

sim::math::Vec2 RotateTowards(const sim::math::Vec2 current, const sim::math::Vec2 desired, const float maxRadians)
{
    if (desired.LengthSquared() <= sim::math::Vec2::kEpsilon) {
        return current;
    }

    if (current.LengthSquared() <= sim::math::Vec2::kEpsilon) {
        return desired.Normalized();
    }

    const float currentAngle = std::atan2(current.y, current.x);
    const float desiredAngle = std::atan2(desired.y, desired.x);
    const float delta = std::clamp(WrapAngle(desiredAngle - currentAngle), -maxRadians, maxRadians);
    const float nextAngle = currentAngle + delta;

    return {std::cos(nextAngle), std::sin(nextAngle)};
}

float BoundaryDistanceAlongRay(const sim::math::Vec2 position,
                               const sim::math::Vec2 direction,
                               const Simulation::PerceivedBoundary& boundary)
{
    if (!boundary.remembered) {
        return kPlayerPerceptionRadius;
    }

    const float denom = direction.Dot(boundary.normal);
    if (denom >= -sim::math::Vec2::kEpsilon) {
        return kPlayerPerceptionRadius;
    }

    const float distanceToPlane = (boundary.start - position).Dot(boundary.normal);
    if (distanceToPlane <= 0.0f) {
        return 0.0f;
    }

    return std::clamp(distanceToPlane / -denom, 0.0f, kPlayerPerceptionRadius);
}

sim::math::Vec2 DirectionFromAngle(const float radians)
{
    return {std::cos(radians), std::sin(radians)};
}

} // namespace

Simulation::Simulation(sim::core::Logger& logger)
    : Simulation(logger, Config{})
{
}

Simulation::Simulation(sim::core::Logger& logger, const Config config)
    : logger_(logger)
    , timer_({.tickRateHz = config.tickRateHz, .realTimePacing = config.realTimePacing})
    , player_({20.0f, 20.0f})
    , config_(config)
{
    player_.SetWantedLevel(1);
    playerKnownBoundaries_ = {
        PerceivedBoundary{{kWorldMin, kWorldMin}, {kWorldMin, kWorldMax}, {1.0f, 0.0f}},
        PerceivedBoundary{{kWorldMax, kWorldMin}, {kWorldMax, kWorldMax}, {-1.0f, 0.0f}},
        PerceivedBoundary{{kWorldMin, kWorldMin}, {kWorldMax, kWorldMin}, {0.0f, 1.0f}},
        PerceivedBoundary{{kWorldMin, kWorldMax}, {kWorldMax, kWorldMax}, {0.0f, -1.0f}}
    };
    policeManager_.SyncToWantedLevel(player_.GetWantedLevel(), player_.GetPosition());
}

void Simulation::Run()
{
    logger_.Info("Simulation started (headless).");

    while (!IsFinished()) {
        Tick();
    }

    logger_.Info("Simulation finished at t=" + std::to_string(timer_.GetElapsedTime()) + "s");
}

void Simulation::UpdateRealtime(const float frameDeltaSeconds)
{
    if (deathPauseRemaining_ <= 0.0f) {
        return;
    }

    deathPauseRemaining_ -= frameDeltaSeconds;
    if (deathPauseRemaining_ <= 0.0f) {
        RespawnPlayer();
    }
}

void Simulation::Tick()
{
    if (IsPaused()) {
        return;
    }

    timer_.Advance();

    const float deltaTime = timer_.GetDeltaTime();
    playerAttackCooldown_ = std::max(0.0f, playerAttackCooldown_ - deltaTime);
    UpdatePlayer(deltaTime);
    const float playerHealthBeforePoliceUpdate = player_.GetHealth();
    UpdatePolice(deltaTime);
    UpdatePursuitEscalation(deltaTime, playerHealthBeforePoliceUpdate);
    TryPlayerAttackPolice(deltaTime);

    if (player_.GetHealth() <= 0.0f && !player_.IsDead()) {
        BeginDeathSequence();
    }
}

bool Simulation::IsFinished() const
{
    return config_.maxTicks > 0 && timer_.GetTickIndex() >= config_.maxTicks;
}

bool Simulation::IsPaused() const
{
    return deathPauseRemaining_ > 0.0f;
}

bool Simulation::IsPlayerDead() const
{
    return player_.IsDead();
}

float Simulation::GetDeathPauseRemaining() const
{
    return deathPauseRemaining_;
}

const std::string& Simulation::GetLastEventMessage() const
{
    return lastEventMessage_;
}

sim::math::Vec2 Simulation::GetHospitalPosition() const
{
    return hospitalPosition_;
}

float Simulation::GetPlayerCurrentSpeed() const
{
    return playerCurrentSpeed_;
}

float Simulation::GetPlayerPerceptionRadius() const
{
    return kPlayerPerceptionRadius;
}

std::size_t Simulation::GetPlayerPerceivedObstacleCount() const
{
    return static_cast<std::size_t>(std::count_if(playerKnownBoundaries_.begin(), playerKnownBoundaries_.end(),
                                                  [](const PerceivedBoundary& boundary) {
                                                      return boundary.remembered;
                                                  }));
}

float Simulation::GetPursuitTimerSeconds() const
{
    return pursuitTimerSeconds_;
}

float Simulation::GetPursuitFailureSeconds() const
{
    return kPursuitFailureDuration;
}

void Simulation::UpdatePlayerPerception(const float deltaTime)
{
    for (PerceivedBoundary& boundary : playerKnownBoundaries_) {
        boundary.visible = false;
        if (boundary.remembered) {
            boundary.lastSeenSeconds += deltaTime;
            if (boundary.lastSeenSeconds > kObstacleMemoryDuration) {
                boundary.remembered = false;
            }
        }
    }

    const sim::math::Vec2 position = player_.GetPosition();
    const float distances[] = {
        position.x - kWorldMin,
        kWorldMax - position.x,
        position.y - kWorldMin,
        kWorldMax - position.y
    };

    for (std::size_t index = 0; index < playerKnownBoundaries_.size(); ++index) {
        if (distances[index] <= kPlayerPerceptionRadius) {
            playerKnownBoundaries_[index].visible = true;
            playerKnownBoundaries_[index].remembered = true;
            playerKnownBoundaries_[index].lastSeenSeconds = 0.0f;
        }
    }
}

void Simulation::UpdatePlayer(const float deltaTime)
{
    if (player_.IsDead()) {
        playerCurrentSpeed_ = 0.0f;
        playerVelocity_ = {};
        playerSteeringVector_ = {};
        return;
    }

    UpdatePlayerPerception(deltaTime);
    playerCurrentSpeed_ = MoveTowards(playerCurrentSpeed_, CalculatePlayerTargetSpeed(), kPlayerAcceleration * deltaTime);

    sim::math::Vec2 dangerEscape;
    playerDangerLevel_ = 0.0f;
    for (const auto& police : policeManager_.GetPoliceNpcs()) {
        if (!police.IsAlive()) {
            continue;
        }

        const sim::math::Vec2 away = player_.GetPosition() - police.GetPosition();
        const float distance = std::max(away.Length(), 1.0f);
        const float weight = std::max(0.0f, (kPoliceDangerRadius - distance) / kPoliceDangerRadius);
        dangerEscape += away.Normalized() * (weight * weight);
        playerDangerLevel_ += weight;
    }

    float bestScore = -1000000.0f;
    sim::math::Vec2 bestDirection = playerCurrentHeading_;
    const float baseAngle = std::atan2(playerCurrentHeading_.y, playerCurrentHeading_.x);
    constexpr int kCandidateCount = 16;

    for (int candidate = 0; candidate < kCandidateCount; ++candidate) {
        const float t = static_cast<float>(candidate) / static_cast<float>(kCandidateCount);
        const float angle = baseAngle + ((t - 0.5f) * 6.28318530718f);
        const sim::math::Vec2 direction = DirectionFromAngle(angle);

        float freeSpace = kPlayerPerceptionRadius;
        for (const PerceivedBoundary& boundary : playerKnownBoundaries_) {
            freeSpace = std::min(freeSpace, BoundaryDistanceAlongRay(player_.GetPosition(), direction, boundary));
        }

        float policeSafety = 0.0f;
        for (const auto& police : policeManager_.GetPoliceNpcs()) {
            if (!police.IsAlive()) {
                continue;
            }

            const sim::math::Vec2 futurePosition = player_.GetPosition() + (direction * kPlayerPerceptionRadius * 0.5f);
            policeSafety += std::min(sim::math::Distance(futurePosition, police.GetPosition()), kPoliceDangerRadius) /
                            kPoliceDangerRadius;
        }

        const sim::math::Vec2 dangerDirection = dangerEscape.Normalized();
        const float continuity = direction.Dot(playerCurrentHeading_);
        const float dangerAlignment = direction.Dot(dangerDirection);
        const float freeSpaceScore = freeSpace / kPlayerPerceptionRadius;
        const float deadEndPenalty = freeSpaceScore < 0.35f && dangerAlignment > 0.0f ? 1.0f : 0.0f;
        const float score = (policeSafety * 2.0f) + (freeSpaceScore * 3.0f) +
                            (dangerAlignment * 1.5f) + (continuity * 0.45f) - deadEndPenalty;

        if (score > bestScore) {
            bestScore = score;
            bestDirection = direction;
        }
    }

    playerEscapeScore_ = bestScore;
    playerEscapeVector_ = ((bestDirection * 0.75f) + (dangerEscape.Normalized() * 0.25f)).Normalized();
    if (playerEscapeVector_.LengthSquared() <= sim::math::Vec2::kEpsilon) {
        playerEscapeVector_ = playerCurrentHeading_;
    }

    playerDesiredHeading_ = playerEscapeVector_;
    playerSteeringVector_ = playerDesiredHeading_ - playerCurrentHeading_;
    playerCurrentHeading_ = RotateTowards(playerCurrentHeading_, playerDesiredHeading_,
                                          kPlayerMaxTurnRateRadiansPerSecond * deltaTime);
    playerVelocity_ = playerCurrentHeading_ * playerCurrentSpeed_;
    player_.Translate(playerVelocity_ * deltaTime);
    player_.SetPosition({
        std::clamp(player_.GetPosition().x, kWorldMin, kWorldMax),
        std::clamp(player_.GetPosition().y, kWorldMin, kWorldMax)
    });

    const auto tick = timer_.GetTickIndex();
    if (tick == 5) {
        player_.IncreaseWantedLevel(2);
        RecordEvent("Player committed a crime — wanted level increased.");
    } else if (tick == 15) {
        player_.IncreaseWantedLevel(1);
        RecordEvent("Player committed another offense — wanted level increased.");
    }
}

void Simulation::UpdatePolice(const float deltaTime)
{
    policeManager_.SyncToWantedLevel(player_.GetWantedLevel(), player_.GetPosition());
    policeManager_.UpdateAll(deltaTime, player_, playerVelocity_);
}

void Simulation::UpdatePursuitEscalation(const float deltaTime, const float playerHealthBeforePoliceUpdate)
{
    const bool policeDamagedPlayer = player_.GetHealth() < playerHealthBeforePoliceUpdate;
    if (player_.IsDead() || policeDamagedPlayer || !policeManager_.AnyInState(sim::entities::NpcState::Pursue)) {
        pursuitTimerSeconds_ = 0.0f;
        return;
    }

    pursuitTimerSeconds_ += deltaTime;
    if (pursuitTimerSeconds_ < kPursuitFailureDuration) {
        return;
    }

    pursuitTimerSeconds_ = 0.0f;

    if (player_.GetWantedLevel() < sim::entities::Player::kMaxWantedLevel) {
        player_.IncreaseWantedLevel(1);
        policeManager_.SyncToWantedLevel(player_.GetWantedLevel(), player_.GetPosition());
        RecordEvent("Pursuit failed for 10s. Wanted level increased.");
    } else {
        RecordEvent("Pursuit failed for 10s. Wanted level is already maxed.");
    }
}

void Simulation::TryPlayerAttackPolice(const float /*deltaTime*/)
{
    if (!player_.IsAlive() || playerAttackCooldown_ > 0.0f) {
        return;
    }

    sim::entities::PoliceNpc* nearestPolice = policeManager_.GetNearestPolice(player_.GetPosition());
    if (nearestPolice == nullptr) {
        return;
    }

    const float distance = sim::math::Distance(player_.GetPosition(), nearestPolice->GetPosition());
    if (distance > kPlayerAttackRange) {
        return;
    }

    nearestPolice->ApplyDamage(kPlayerAttackDamage);
    playerAttackCooldown_ = kPlayerAttackCooldown;
    RecordEvent("Player counterattacked police.");
}

float Simulation::CalculatePlayerTargetSpeed() const
{
    const sim::entities::PoliceNpc* nearestPolice = policeManager_.GetNearestPolice(player_.GetPosition());
    if (nearestPolice == nullptr) {
        return kPlayerBaseSpeed;
    }

    const float distance = sim::math::Distance(player_.GetPosition(), nearestPolice->GetPosition());
    if (distance >= kPlayerAccelerationDistance) {
        return kPlayerBaseSpeed;
    }

    const float closeness = 1.0f - (distance / kPlayerAccelerationDistance);
    return kPlayerBaseSpeed + ((kPlayerMaxSpeed - kPlayerBaseSpeed) * closeness);
}

void Simulation::BeginDeathSequence()
{
    player_.MarkDead();
    deathPauseRemaining_ = kDeathPauseDuration;
    RecordEvent("PLAYER DIED");
}

void Simulation::RespawnPlayer()
{
    player_.Respawn(hospitalPosition_, sim::entities::Player::kMaxHealth);
    player_.SetWantedLevel(kRespawnWantedLevel);
    policeManager_.SyncToWantedLevel(player_.GetWantedLevel(), player_.GetPosition());
    deathPauseRemaining_ = 0.0f;
    RecordEvent("Player respawned at hospital. Wanted level restored to 3 stars.");
}

void Simulation::RecordEvent(std::string message)
{
    lastEventMessage_ = std::move(message);
    logger_.Warn(lastEventMessage_);
}

} // namespace sim
